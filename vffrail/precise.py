"""PRECISE traces at junctions from the already computed probabilities (no model, no GPU).

"Junctions within 3 m" work package (2026-10-20), final requirement:
absolute accuracy p90 <= 5 m on the kept pieces, 0 errors beyond 10 m, and ABSOLUTE continuity at the
junction (the branch ends EXACTLY on the geometry of the track it joins, tangentially).

Pipeline (local "ground" frame: EPSG:3857 coordinates divided by the Mercator factor s = 1/cos(lat)):
1. LOCAL registration of the orthophoto georeferencing: the precise tracks (BD TOPO / OSM) visible in the same
   image give, through transverse probability profiles, the image - real offset (translation smoothed along the
   branch, robust least squares, covariance);
2. sub-pixel axis: transverse probability profiles along the extracted path, centre of the half-height plateau
   (the model is trained on a ± 3 m band: a plateau, not a peak);
3. railway geometry: penalised fit (third difference: continuous curvature, piecewise linear =
   straights, arcs, clothoids), robust least squares, radius check (minRadius of the engine);
4. switch: where the branch is within SEP_M of the partner track (tracks merged in the image), the
   branch is replaced by a turnout curve TANGENT to the partner track at the switch point S (searched
   along the partner track by fitting to the probability), joined tangentially to the fitted axis; S is a
   point of the partner geometry (shared vertex, to be inserted in the precise track);
5. estimated error per station (registration, fit, ambiguity, switch model) and strict filter.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import shapely
import shapely.ops
from shapely import LineString

from .geo import TILE, merc_scale, merc_to_lonlat, res

Z = 16
STEP_M = 2.0            # station spacing along the axis
SEP_M = 18.0            # separation below which the plateaus of two tracks merge (plateau 14-24 m)
REG_HALF_M = 30.0       # half-window of the registration profiles
AXIS_HALF_M = 8.0       # search for the axis peak around the extracted path
PROF_HALF_M = 30.0      # half-window of the axis profiles
MAX_W_M = 36.0          # maximum width of a closed plateau (at half height; double track, station: up to ~33 m)
OBS_W_M = 22.0          # plateau narrow enough for its centre to be the axis (beyond: station, multiple tracks)
REGISTER = False        # local registration measured with no gain on the control set: disabled
MIN_CUT_M = 20.0        # minimum merged zone to cut the branch
MIN_PIECE_M = 40.0
P_MIN = 0.35            # minimum probability of a plateau
REG_KERNEL_M = 600.0    # range (standard deviation) of the local registration
REG_PRIOR_M = 8.0       # prior standard deviation of the orthophoto shift (unobserved direction)
LAMBDA = 2.0e4          # weight of the third-difference penalty (smoothing ~ 60 m)
HUBER_M = 1.5
S_SEARCH_M = 300.0
COARSE_MAX_M = 20.0     # maximum offset allowed from the coarse trace (beyond: another track is likely)
AMB_EXPLAINED_M = 5.0   # another plateau within 5 m of a known precise track is "explained"


def min_radius(network: str | None, gauge: int | None) -> float:
    """Minimum plausible radius (m), identical to JunctionRepair::minRadius of the engine."""
    if (gauge is not None and gauge <= 600) or (gauge is None and network == 'militaire'):
        return 25.0
    if (gauge is not None and gauge < 1435) or (gauge is None and network == 'secondaire'):
        return 60.0
    if network == 'principal':
        return 190.0
    return 150.0


# ===== probabilities ================================================================


class Mosaic:
    """Probabilities (z16) of an extent, in the local ground frame; missing tiles = NaN."""

    def __init__(self, prob_root, bbox_merc, lat: float, z: int = Z, arr=None, origin_px=None):
        self.z, self.s = z, merc_scale(lat)
        self.r = res(z)
        if arr is not None:  # tests
            self.P, (self.ox, self.oy) = np.asarray(arr, np.float32), origin_px
            return
        from PIL import Image

        x0, y0, x1, y1 = bbox_merc
        R = 20037508.342789244
        tx0, tx1 = int((x0 + R) / self.r // TILE), int((x1 + R) / self.r // TILE)
        ty0, ty1 = int((R - y1) / self.r // TILE), int((R - y0) / self.r // TILE)
        P = np.full(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE), np.nan, np.float32)
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                p = Path(prob_root) / str(z) / str(tx) / f'{ty}.png'
                if p.exists():
                    P[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = \
                        np.asarray(Image.open(p), np.float32) / 255.0
        self.P, self.ox, self.oy = P, tx0 * TILE, ty0 * TILE

    @property
    def gres(self) -> float:
        return self.r / self.s

    def to_px(self, xy_g):
        """Local ground frame (merc / s) -> (row, column) in the mosaic."""
        R = 20037508.342789244
        xy = np.asarray(xy_g, float) * self.s
        col = (xy[..., 0] + R) / self.r - self.ox
        row = (R - xy[..., 1]) / self.r - self.oy
        return row, col

    def sample(self, xy_g):
        from scipy import ndimage

        row, col = self.to_px(xy_g)
        shp = row.shape
        if getattr(self, '_filled', None) is None:
            self._filled = np.nan_to_num(self.P, nan=-1.0)
        v = ndimage.map_coordinates(self._filled, [row.ravel() - 0.5, col.ravel() - 0.5], order=1, mode='constant', cval=-1.0)
        v = v.reshape(shp)
        return np.where(v < 0, np.nan, v)


def plateau_centres(prof: np.ndarray, u: np.ndarray, centre_win: float, p_min=P_MIN):
    """Sub-pixel centre of the probability plateau closest to 0 within [-centre_win, centre_win].

    prof: (n, k) profiles sampled at abscissas u (m, increasing). Returns (centre, peak, width, ambiguous):
    centre = centroid of (p - peak/2)+ over the connected part around the peak; ambiguous = another plateau of height
    >= 0.6 peak in the full window, separated from the first by a dip below half height.
    """
    n, k = prof.shape
    out = np.full((n, 4), np.nan)
    inner = np.abs(u) <= centre_win
    for i in range(n):
        p = prof[i]
        if np.isnan(p[inner]).any():
            continue
        pin = np.where(inner, p, -1.0)
        j = int(np.argmax(pin))
        pk = p[j]
        if pk < p_min:
            continue
        half = pk / 2.0
        a = j
        while a > 0 and not np.isnan(p[a - 1]) and p[a - 1] >= half:
            a -= 1
        b = j
        while b < k - 1 and not np.isnan(p[b + 1]) and p[b + 1] >= half:
            b += 1
        if a == 0 or b == k - 1:  # plateau not closed within the window: centre unknown
            continue
        w = p[a:b + 1] - half
        c = float((u[a:b + 1] * w).sum() / w.sum())
        rest = np.concatenate([p[:max(0, a - 1)], p[b + 2:]])
        amb = bool(np.nanmax(rest) >= 0.6 * pk) if len(rest) and not np.isnan(rest).all() else False
        out[i] = (c, pk, u[b] - u[a], float(amb))
    return out


def other_plateaus(prof: np.ndarray, u: np.ndarray, main_c: np.ndarray, rel=0.6):
    """Centres (m) of the OTHER plateaus of a profile (height >= rel x main peak), per station (list)."""
    out = []
    for i in range(len(prof)):
        p, c = prof[i], main_c[i]
        if np.isnan(c) or np.isnan(p).all():
            out.append([])
            continue
        pk = np.nanmax(p[np.abs(u - c) <= 3.0]) if (np.abs(u - c) <= 3.0).any() else np.nan
        hi = np.nan_to_num(p, nan=0.0) >= max(P_MIN, pk / 2.0)
        idx = np.flatnonzero(np.diff(np.concatenate([[0], hi.astype(int), [0]])))
        cs = []
        for a, b in zip(idx[::2], idx[1::2]):
            if b - a < 2 or a == 0 or b == len(p):
                continue
            seg = p[a:b]
            cc = float((u[a:b] * (seg - pk / 2)).sum() / max(1e-9, (seg - pk / 2).sum()))
            if abs(cc - c) > 4.0 and seg.max() >= rel * pk:
                cs.append(cc)
        out.append(cs)
    return out


def profiles(mos: Mosaic, pts, normals, half_m: float, du: float = 0.4):
    u = np.arange(-half_m, half_m + 1e-9, du)
    xy = pts[:, None, :] + u[None, :, None] * normals[:, None, :]
    return mos.sample(xy), u


# ===== geometry =====================================================================


def resample(coords, step=STEP_M):
    c = np.asarray(coords, float)
    seg = np.hypot(*np.diff(c, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return c[:1]
    n = max(2, int(round(cum[-1] / step)) + 1)
    sv = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(sv, cum, c[:, 0]), np.interp(sv, cum, c[:, 1])], 1)


def tangents(p):
    t = np.gradient(p, axis=0)
    return t / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]


def curvature(p):
    """Signed curvature (1/m) of a regularly spaced polyline."""
    if len(p) < 3:
        return np.zeros(len(p))
    h = np.unwrap(np.arctan2(*np.gradient(p, axis=0).T[::-1]))
    ds = np.hypot(*np.gradient(p, axis=0).T)
    return np.gradient(h) / np.maximum(ds, 1e-9)


def max_turn_rate(p, win_m=20.0):
    """Largest heading change (rad/m) over a window of win_m (detects bends, insensitive to noise)."""
    if len(p) < 3:
        return 0.0
    q = resample(p, 1.0)
    k = int(win_m)
    if len(q) <= k + 1:
        k = len(q) - 2
    if k < 2:
        return 0.0
    h = np.unwrap(np.arctan2(*np.diff(q, axis=0).T[::-1]))
    return float(np.max(np.abs(h[k:] - h[:-k])) / k)


def fit_rail(ref, obs_off, obs_w, lam=LAMBDA, huber=HUBER_M, iters=6):
    """Railway fit: p_i = ref_i + w_i n_i (lateral offset), minimising
    Σ ρ(w_i - o_i) + lam Σ (Δ³w)² + a small tie to 0. Returns (fitted points, offsets, residuals)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n = len(ref)
    nrm = tangents(ref) @ np.array([[0.0, 1.0], [-1.0, 0.0]])  # left normal
    o = np.nan_to_num(obs_off)
    w0 = np.where(np.isnan(obs_off), 0.0, obs_w)
    if n < 5:
        return ref + o[:, None] * nrm, o, np.zeros(n)
    D = sparse.diags([-1.0, 3.0, -3.0, 1.0], [0, 1, 2, 3], shape=(n - 3, n))
    Q = lam * (D.T @ D)  # constant step: third-difference penalty
    rw = np.ones(n)
    off = o.copy()
    for _ in range(iters):
        W = sparse.diags(w0 * rw + 1e-6)
        off = spsolve((W + Q).tocsc(), w0 * rw * o)
        r = off - o
        rw = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-9))
    return ref + off[:, None] * nrm, off, np.where(w0 > 0, off - o, np.nan)


# ===== local registration ===========================================================


def registration_obs(mos: Mosaic, tracks_g, avoid_g=(), step=5.0, half=REG_HALF_M):
    """Image - real offset observations on precise tracks: (x, y, nx, ny, d).

    A station is ignored if another track (precise or to be avoided: the branch itself) passes within half m
    (merged plateaus); d = position of the plateau along the normal (ground m)."""
    rows = []
    geoms = [LineString(t) for t in tracks_g if len(t) >= 2]
    av = [LineString(a) for a in avoid_g if len(a) >= 2]
    for gi, g in enumerate(geoms):
        if g.length < 20:
            continue
        p = resample(np.asarray(g.coords), step)
        nrm = tangents(p) @ np.array([[0.0, 1.0], [-1.0, 0.0]])
        others = [h for hj, h in enumerate(geoms) if hj != gi and h.distance(g) < half + 2] + av
        ok = np.ones(len(p), bool)
        pts = shapely.points(p)
        for h in others:
            # another track in the transverse window, unless it coincides with this one (same axis within 1 m)
            d = shapely.distance(pts, h)
            ok &= ~((d < half + 2) & (d > 1.0))
        if not ok.any():
            continue
        prof, u = profiles(mos, p[ok], nrm[ok], half)
        pc = plateau_centres(prof, u, 8.0)
        good = ~np.isnan(pc[:, 0]) & (pc[:, 2] <= MAX_W_M) & (pc[:, 3] < 0.5)
        q, nq = p[ok][good], nrm[ok][good]
        for (x, y), (nx, ny), d in zip(q, nq, pc[good, 0]):
            rows.append((x, y, nx, ny, d))
    return np.asarray(rows, float).reshape(-1, 5)


def solve_shift(obs, at_xy, kernel=REG_KERNEL_M, prior=REG_PRIOR_M, huber=2.0):
    """Image - real translation at the point at_xy (robust least squares weighted by distance).

    Returns (t (2,), covariance (2, 2), residual sd, effective n)."""
    if not len(obs):
        return np.zeros(2), np.eye(2) * prior ** 2, None, 0.0
    d2 = ((obs[:, :2] - at_xy) ** 2).sum(1)
    ws = np.exp(-0.5 * d2 / kernel ** 2)
    N, dd = obs[:, 2:4], obs[:, 4]
    t = np.zeros(2)
    rw = np.ones(len(obs))
    sd = 1.5
    for _ in range(8):
        w = ws * rw
        A = (N * w[:, None]).T @ N / sd ** 2 + np.eye(2) / prior ** 2
        b = (N * (w * dd)[:, None]).sum(0) / sd ** 2
        t = np.linalg.solve(A, b)
        r = dd - N @ t
        rw = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-9))
        neff = float(ws.sum())
        if neff > 3:
            sd = max(0.5, float(np.sqrt((ws * rw * r ** 2).sum() / max(1e-9, (ws * rw).sum()))))
    A = (N * (ws * rw)[:, None]).T @ N / sd ** 2 + np.eye(2) / prior ** 2
    neff = float((ws * rw).sum())
    return t, np.linalg.inv(A), sd, neff


# ===== switch =======================================================================


def hermite(S, d0, Q, d1, m, n=None):
    L = float(np.hypot(*(Q - S)))
    if n is None:
        n = max(8, int(L / 1.0))
    t = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * S + h10 * (m * d0) + h01 * Q + h11 * (m * d1)


def switch_leg(G: LineString, Q, uQ, mos: Mosaic | None, shift, rmin: float, sep_m=SEP_M, step=0.5,
               search=S_SEARCH_M, score_band=(4.0, 12.0)):
    """Switch curve from the partner track G (ground frame, real) to the point Q of the branch (real).

    uQ: tangent of the branch at Q, oriented TOWARDS the junction. The curve starts at S ∈ G, tangent to G, and arrives
    at Q tangent to -uQ. S is chosen by the mean probability along the curve (image = real + shift) within the
    separation band score_band; constraints: increasing separation, radius >= rmin, no bend.
    Returns dict(S, curve, sigma_s, e_model, score, n_ok) or None."""
    Q = np.asarray(Q, float)
    d1 = -np.asarray(uQ, float)
    sq = G.project(shapely.Point(Q))
    lo, hi = max(0.0, sq - search), min(G.length, sq + search)
    g0 = max(0.0, lo - 80)
    Gl = shapely.ops.substring(G, g0, min(G.length, hi + 80))
    eobs = None
    if mos is not None:
        es, ew = edge_obs(mos, G, Q, shift, lo - 60, hi + 60)
        eobs = (es, ew, g0)
    cands = _leg_cands(G, Gl, Q, d1, mos, shift, rmin, sep_m, np.arange(lo, hi + 1e-9, 4.0), score_band, eobs)
    if cands:
        s0 = max(cands, key=lambda x: x[0])[1]
        cands += _leg_cands(G, Gl, Q, d1, mos, shift, rmin, sep_m, np.arange(max(lo, s0 - 4), min(hi, s0 + 4) + 1e-9, step),
                            score_band, eobs)
    if not cands:
        return None
    return _best_leg(G, cands)


def edge_obs(mos: Mosaic, G: LineString, side_pt, shift, sig_lo, sig_hi, step=STEP_M, half_out=40.0, half_in=25.0):
    """Offset of the branch from the track G deduced from the EDGE of the common plateau (zone where the two tracks merge).

    Along G (real; image = real + shift), normal profile oriented towards the branch (side of side_pt):
    outer edge e+ of the plateau containing G, inner edge e- (opposite side) = own half-width of G.
    Observation w = e+ - |e-| (0 if the branch has not yet left the plateau of G).
    Returns (sigma, w); NaN if the profile is unusable (no plateau on G, edges not closed)."""
    sig = np.arange(max(0.0, sig_lo), min(G.length, sig_hi) + 1e-9, step)
    if len(sig) < 3:
        return sig, np.full(len(sig), np.nan)
    pts = np.asarray([G.interpolate(x).coords[0] for x in sig])
    tg = tangents(pts) if len(pts) > 2 else np.tile((pts[-1] - pts[0]) / max(1e-9, np.hypot(*(pts[-1] - pts[0]))), (len(pts), 1))
    nrm = tg @ np.array([[0.0, 1.0], [-1.0, 0.0]])
    sp = np.asarray(side_pt, float)
    k0 = int(np.argmin(np.hypot(*(pts - sp).T)))
    if nrm[k0] @ (sp - pts[k0]) < 0:
        nrm = -nrm
    u = np.arange(-half_in, half_out + 1e-9, 0.4)
    prof = mos.sample(pts[:, None, :] + shift + u[None, :, None] * nrm[:, None, :])
    w = np.full(len(sig), np.nan)
    i0 = int(np.argmin(np.abs(u)))
    for i in range(len(sig)):
        p = prof[i]
        if np.isnan(p).any():
            continue
        win = np.abs(u) <= 4.0
        j = int(np.argmax(np.where(win, p, -1.0)))
        pk = p[j]
        if pk < P_MIN:
            continue
        half = pk / 2.0
        a = j
        while a > 0 and p[a - 1] >= half:
            a -= 1
        b = j
        while b < len(p) - 1 and p[b + 1] >= half:
            b += 1
        if a == 0 or b == len(p) - 1:
            continue
        e_in, e_out = -u[a], u[b]
        if not (2.0 <= e_in <= MAX_W_M / 2 + 2):
            continue  # opposite side cluttered (another track): half-width unknown
        w[i] = max(0.0, e_out - e_in)
    del i0
    return sig, w


def _offset_profile(c, G, sig):
    """Lateral offset (m) of a curve c from G, at the abscissas sig of G (0 outside the extent of the curve, before S)."""
    sc = shapely.line_locate_point(G, shapely.points(c))
    dc = shapely.distance(shapely.points(c), G)
    o = np.argsort(sc)
    sc, dc = sc[o], dc[o]
    w = np.interp(sig, sc, dc, left=np.nan, right=np.nan)
    return w


def _leg_cands(G, Gl, Q, d1, mos, shift, rmin, sep_m, sigmas, score_band, eobs=None):
    cands = []
    for sig in sigmas:
        S = np.asarray(G.interpolate(sig).coords[0])
        a, b = G.interpolate(max(0.0, sig - 2.0)), G.interpolate(min(G.length, sig + 2.0))
        tau = np.asarray(b.coords[0]) - np.asarray(a.coords[0])
        nt = np.hypot(*tau)
        if nt < 1e-6:
            continue
        tau /= nt
        ch = Q - S
        L = float(np.hypot(*ch))
        if L < sep_m * 1.5:
            continue
        d0 = tau if tau @ ch > 0 else -tau
        if d0 @ ch / L < math.cos(math.radians(30)):
            continue
        for lam in (0.7, 1.0, 1.3):
            c = hermite(S, d0, Q, d1, lam * L)
            if max_turn_rate(c, min(20.0, L / 3)) > 1.0 / rmin:
                continue
            sep = shapely.distance(shapely.points(c), Gl)
            if np.any(np.diff(sep) < -0.05) or sep[-1] < sep_m - 1.5:
                continue
            if eobs is not None and np.isfinite(eobs[1]).sum() >= 5:
                es, ew = eobs[0], eobs[1]
                wm = _offset_profile(c, Gl, es - eobs[2])
                d_s = shapely.line_locate_point(G, shapely.Point(c[0]))
                wm = np.where(es < d_s, 0.0, wm)
                ok = np.isfinite(ew) & np.isfinite(wm) & ((wm <= 16.0) | (ew > 0.5))
                if ok.sum() < 5:
                    continue
                r = np.abs(wm[ok] - ew[ok])
                rho = np.where(r <= HUBER_M, 0.5 * r ** 2, HUBER_M * (r - 0.5 * HUBER_M))
                sc = -float(rho.mean())
            else:
                sc = 0.0
            cands.append((sc, sig, lam, c))
    return cands


def _best_leg(G, cands):
    cands.sort(key=lambda x: -x[0])
    sc, sig, lam, c = cands[0]
    # model uncertainty: maximum lateral offset to the curves with a close score (within 0.05 of the best)
    near = [x for x in cands if x[0] >= sc - max(0.05, 0.25 * abs(sc))]
    # LATERAL offset (to the chosen curve or to the partner track: before S, the branch is on the partner track)
    bl = shapely.union_all([LineString(c), G])
    e_model = max(float(shapely.distance(shapely.points(x[3]), bl).max()) for x in near)
    sig_s = float(np.std([x[1] for x in near]))
    S = np.asarray(G.interpolate(sig).coords[0])
    c[0] = S
    return {'S': S, 's_on_G': float(sig), 'curve': c, 'score': sc, 'e_model': e_model, 'sigma_s': sig_s,
            'n_ok': len(cands), 'lam': lam}


def _close_runs(mask, min_len, value):
    """Flips the runs of `not value` shorter than min_len stations to `value` (fills small gaps)."""
    m = np.asarray(mask, bool).copy()
    x = (m != value).astype(int)
    idx = np.flatnonzero(np.diff(np.concatenate([[0], x, [0]])))
    for a, b in zip(idx[::2], idx[1::2]):
        if b - a < min_len:
            m[a:b] = value
    return m


# ===== one target ===================================================================


def process_target(target: dict, feats: list[dict], mos: Mosaic, partners_g: list[dict], reg_tracks_g: list,
                   rmin: float, truth_g=None, sep_m=SEP_M, other_g=None, coarse_g=None, coarse_max_m=COARSE_MAX_M):
    """One junction target -> precise branches (ground frame). feats: extracted paths (ground frame, image).

    partners_g: partner tracks [{'g': ground coords, 'line_id', 'part_id', 'src'}]; reg_tracks_g: precise tracks
    used for the registration. Returns a list of branches {pts, e, flags, ends: [junctions]}."""
    out = []
    obs = registration_obs(mos, reg_tracks_g, avoid_g=[f for f in feats])
    G = [(LineString(p['g']), p) for p in partners_g if len(p['g']) >= 2]
    others = [LineString(g) for g in (other_g if other_g is not None else [p['g'] for p in partners_g]) if len(g) >= 2]
    OT = shapely.MultiLineString(others) if others else None
    CO = LineString(np.asarray(coarse_g)) if coarse_g is not None else None
    for fi, fc in enumerate(feats):
        p0 = resample(fc, STEP_M)
        if len(p0) < 10:
            continue
        # 1. local registration along the path (anchors every 250 m, interpolation)
        cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p0, axis=0).T))])
        anchors = np.unique(np.concatenate([np.arange(0, cum[-1], 250.0), [cum[-1]]]))
        sh = []
        for a in anchors:
            i = int(np.searchsorted(cum, a).clip(0, len(p0) - 1))
            sh.append(solve_shift(obs, p0[i]))
        tx = np.interp(cum, anchors, [x[0][0] for x in sh])
        ty = np.interp(cum, anchors, [x[0][1] for x in sh])
        shift = np.stack([tx, ty], 1) if REGISTER else np.zeros((len(p0), 2))
        # 2. sub-pixel axis in the image
        nrm = tangents(p0) @ np.array([[0.0, 1.0], [-1.0, 0.0]])
        prof, u = profiles(mos, p0, nrm, PROF_HALF_M)
        pc = plateau_centres(prof, u, AXIS_HALF_M)
        # separation from the partner tracks (real): where < sep_m, the tracks merge -> no observation
        real0 = p0 - shift
        sep = np.full(len(p0), np.inf)
        part_of = np.full(len(p0), -1)
        for gi, (g, _) in enumerate(G):
            d = shapely.distance(shapely.points(real0), g)
            better = d < sep
            sep[better], part_of[better] = d[better], gi
        valid = ~np.isnan(pc[:, 0]) & (pc[:, 2] <= OBS_W_M) & (sep >= sep_m)
        o = np.where(valid, pc[:, 0], np.nan)
        wgt = np.where(valid, np.clip(pc[:, 1], 0, 1) * np.where(pc[:, 3] > 0.5, 0.3, 1.0), 0.0)
        img_fit, off, resid = fit_rail(p0, o, wgt)
        real = img_fit - shift
        # 3. separation recomputed on the fitted axis; split into free pieces (>= sep_m from any partner track)
        sep = np.full(len(real), np.inf)
        part_of = np.full(len(real), -1)
        for gi, (g, _) in enumerate(G):
            d = shapely.distance(shapely.points(real), g)
            better = d < sep
            sep[better], part_of[better] = d[better], gi
        free = sep >= sep_m
        if OT is not None:  # precise track of another line (not a partner) merged: no axis of its own
            free &= shapely.distance(shapely.points(real), OT) >= sep_m
        # other plateau not explained by a known track (service track, line missing from the database): ambiguity
        oth = other_plateaus(prof, u, pc[:, 0])
        amb = np.zeros(len(real), bool)
        for k, cs in enumerate(oth):
            for cc in cs:
                q = shapely.Point(*(real[k] + (cc - (off[k] if np.isfinite(off[k]) else 0.0)) * nrm[k]))
                if OT is None or q.distance(OT) > AMB_EXPLAINED_M:
                    if not any(q.distance(g) <= AMB_EXPLAINED_M for g, _ in G):
                        amb[k] = True
        dco = shapely.distance(shapely.points(real), CO) if CO is not None else np.zeros(len(real))
        free = _close_runs(free, int(MIN_CUT_M / STEP_M), True)
        free = _close_runs(free, int(MIN_PIECE_M / STEP_M), False)
        idx = np.flatnonzero(np.diff(np.concatenate([[0], free.astype(int), [0]])))
        # errors per station
        cov = [x[1] for x in sh]
        sd_obs = np.array([x[2] if x[2] is not None else np.nan for x in sh])
        nrm_f = tangents(real) @ np.array([[0.0, 1.0], [-1.0, 0.0]])
        var_reg = np.zeros(len(real))
        for k in range(len(real)):
            j = int(np.clip(np.searchsorted(anchors, cum[k]), 0, len(anchors) - 1))
            C = cov[j]
            var_reg[k] = nrm_f[k] @ C @ nrm_f[k] if REGISTER else 1.0
        # fit: local robust residual (± 20 m) and observation coverage
        win = int(20 / STEP_M)
        r2 = np.where(np.isnan(resid), np.nan, np.minimum(resid ** 2, 25.0))
        loc_r = np.full(len(real), np.nan)
        loc_cov = np.zeros(len(real))
        for k in range(len(real)):
            a, b = max(0, k - win), min(len(real), k + win + 1)
            rr = r2[a:b]
            loc_cov[k] = float((~np.isnan(rr)).mean())
            loc_r[k] = float(np.sqrt(np.nanmean(rr))) if (~np.isnan(rr)).any() else np.nan
        for a, b in zip(idx[::2], idx[1::2]):
            seg = real[a:b]
            if len(seg) < 3:
                continue
            e_fit = np.where(np.isnan(loc_r[a:b]), 5.0, loc_r[a:b]) / np.sqrt(np.maximum(1.0, loc_cov[a:b] * 2 * win))
            gap = np.where(loc_cov[a:b] < 0.5, 3.0 * (0.5 - loc_cov[a:b]) * 2, 0.0) ** 2
            e = np.sqrt(var_reg[a:b] + e_fit ** 2 + gap)
            # ambiguities (plausible other track) and offset from the coarse trace: gross error possible -> not precise
            from scipy import ndimage
            amb_w = ndimage.uniform_filter1d(amb[a:b].astype(float), size=2 * win + 1, mode='nearest') > 0.2
            e = np.where(amb_w | (dco[a:b] > coarse_max_m), np.maximum(e, 99.0), e)
            br = {'pts': seg, 'e': e, 'cov': loc_cov[a:b], 'dco': dco[a:b], 'w': pc[a:b, 2], 'pk': pc[a:b, 1], 'amb': amb_w,
                  'fres': loc_r[a:b], 'feat': fi, 'dot': shapely.distance(shapely.points(seg), OT) if OT is not None else np.full(len(seg), 999.0), 'reg_n': [x[3] for x in sh], 'reg_sd': np.nanmedian(sd_obs) if
                  np.isfinite(sd_obs).any() else None, 'turn': max_turn_rate(seg), 'ends': [], 'shift_med':
                  np.median(shift[a:b], 0)}
            # 4. switches at the ends cut near a partner track
            for end, k_in, k_out in (('debut', a, a - 1), ('fin', b - 1, b)):
                if k_out < 0 or k_out >= len(real):
                    continue
                gi = int(part_of[k_out])
                if gi < 0 or sep[k_out] >= sep_m:
                    continue  # cut due to a track other than the partner track: no junction here
                g, pinfo = G[gi]
                Q = real[k_in]
                tq = tangents(seg)[0 if end == 'debut' else -1]
                uQ = -tq if end == 'debut' else tq
                leg = switch_leg(g, Q, uQ, mos, shift[k_in], rmin, sep_m)
                br['ends'].append({'bout': end, 'leg': leg, 'partner': pinfo, 'Q': Q})
            out.append(br)
    if truth_g is not None:
        T = shapely.MultiLineString([LineString(t) for t in truth_g if len(t) >= 2])
        ends = shapely.multipoints([pt for t in truth_g if len(t) >= 2 for pt in (t[0], t[-1])])

        def _err(pts, T=T, ends=ends):
            q = shapely.points(pts)
            d = shapely.distance(q, T)
            # station outside the extent of the truth (closest point = an end of the truth): not evaluable
            near = shapely.get_coordinates(shapely.shortest_line(q, T))[1::2]
            at_end = shapely.distance(shapely.points(near), ends) < 0.5
            d = np.where(at_end & (d > 3.0), np.nan, d)
            if CO is not None:  # truth missing here (coarse trace more than 25 m from any truth): not evaluable
                cq = shapely.get_coordinates(shapely.shortest_line(q, CO))[1::2]
                d = np.where(shapely.distance(shapely.points(cq), T) > 25.0, np.nan, d)
            return d

        for br in out:
            br['err'] = _err(br['pts'])
            for en in br['ends']:
                if en['leg'] is not None:  # truth of the branch + partner track (the branch starts from it)
                    gp = LineString(en['partner']['g'])
                    TU = shapely.union_all([T, gp])
                    en['err'] = _err(en['leg']['curve'], TU, shapely.multipoints([pt for t in truth_g if len(t) >= 2 for pt in (t[0], t[-1])] + [gp.coords[0], gp.coords[-1]]))
    return out, obs


def assemble(br: dict, step=STEP_M):
    """Final geometry of a branch: [S, switch curve, fitted axis, curve, S'] (ground frame).
    The first / last vertex is EXACTLY the switch point on the partner track."""
    pts = br['pts']
    head, tail = None, None
    for en in br['ends']:
        if en['leg'] is None:
            continue
        c = en['leg']['curve']
        if en['bout'] == 'debut':
            head = c[:-1]  # c goes from S to Q = pts[0]
        else:
            tail = c[::-1][1:]  # from Q to S
    parts = [x for x in (head, pts, tail) if x is not None]
    return np.vstack(parts)


def to_lonlat(pts_g, s):
    m = np.asarray(pts_g) * s
    lon, lat = merc_to_lonlat(m[:, 0], m[:, 1])
    return np.stack([lon, lat], 1)
