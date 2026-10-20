"""Post-processing of the pieces extracted around junctions (pure functions, tested without GPU).

On photographs, a junction is only certain to within ~15 m: the track extracted from the imprecise
branch is STOPPED at the first approach within `tol` (15 m) of the precise partner track (BD TOPO / OSM /
registered). The last vertex is not stuck onto the track: the junction step draws the smoothed tangent
connection from this vertex and the attributes written here (interface agreed on 2026-10-20):

    properties.jonction = {jid, kind, partner_line_id, partner_source, raccord: [lon, lat] | None,
                           raccord_ecart_m, cap_voie_deg, cap_branche_deg, tolerance_m, precision_m, raccords: [...]}

Rules:
- the parts within `tol` of a partner track are removed (two tracks cannot be told apart there);
- an end cut this way carries a connection (projection on the partner track, headings);
- a kept piece is at least 50 m long if it carries a connection (starting heading estimable), 200 m otherwise;
- an end that does not approach a partner track has no connection (the branch stays on the coarse trace).
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import LineString, MultiLineString

TOL_M = 15.0
MIN_RACCORD_M = 50.0
MIN_FREE_M = 200.0
CAP_M = 50.0
STEP_M = 2.0


def heading(p, q) -> float:
    """Geographic heading (0 = north, clockwise, degrees) from p to q, Mercator (conformal) coordinates."""
    return (math.degrees(math.atan2(q[0] - p[0], q[1] - p[1])) + 360.0) % 360.0


def axis_heading(line: LineString, pt, near_heading: float, s: float) -> float:
    """Heading of the track `line` at the point closest to `pt`, oriented as close as possible to `near_heading`."""
    a = line.project(shapely.Point(pt))
    d = 10.0 * s
    p0 = line.interpolate(max(0.0, a - d))
    p1 = line.interpolate(min(line.length, a + d))
    h = heading((p0.x, p0.y), (p1.x, p1.y))
    alt = (h + 180.0) % 360.0
    diff = lambda x: abs((x - near_heading + 180.0) % 360.0 - 180.0)  # noqa: E731
    return h if diff(h) <= diff(alt) else alt


def partners_for(target: dict) -> list[dict]:
    """Precise "partner" tracks: lines of the corridor junctions (other line), the line itself
    for a 'meme-ligne' (same line) junction, all neighbouring precise tracks for a detector place."""
    own = target['line_id']
    want, same, anyl = set(), False, False
    for j in target.get('junctions', []):
        want.update(j.get('partners') or [])
        same |= j.get('relation') == 'meme-ligne'
        anyl |= j.get('kind') == 'detecteur'
    out = []
    for p in target.get('precise', []):
        if anyl or p['line_id'] in want or (same and p['line_id'] == own):
            out.append(p)
    return out


def _densify(coords: np.ndarray, step: float) -> np.ndarray:
    seg = np.diff(coords, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(L)])
    if cum[-1] <= 0:
        return coords
    sv = np.unique(np.concatenate([np.arange(0, cum[-1], step), cum]))
    return np.stack([np.interp(sv, cum, coords[:, 0]), np.interp(sv, cum, coords[:, 1])], 1)


def cut_at_partners(coords_merc, s: float, partners: list[dict], junction_pts=(), tol_m=TOL_M,
                    min_raccord_m=MIN_RACCORD_M, min_free_m=MIN_FREE_M):
    """Cuts a piece (EPSG:3857, `s` = Mercator m per ground m) where it comes within tol_m of the partner tracks.

    Returns a list of (coords, connections); connections = list of dicts (end 'debut' | 'fin', Mercator
    connection point, offset m, headings, partner line / source, distance to the closest junction point).
    """
    c = np.asarray(coords_merc, float)
    if len(c) < 2:
        return []
    if not partners:
        L = LineString(c).length / s
        return [(c, [])] if L >= min_free_m else []
    geoms = [LineString(p['coords']) for p in partners]
    d = _densify(c, STEP_M * s)
    pts = shapely.points(d)
    dist = np.stack([shapely.distance(pts, g) for g in geoms], 1) / s   # (n, partners)
    dmin = dist.min(1)
    free = dmin > tol_m
    out = []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], free.astype(int), [0]])))
    jp = [shapely.Point(p) for p in junction_pts]
    for a, b in zip(idx[::2], idx[1::2]):
        # the piece stops at the FIRST point within <= tol of the partner track (included)
        a2, b2 = max(0, a - 1), min(len(d), b + 1)
        seg = d[a2:b2]
        if len(seg) < 2:
            continue
        L = float(np.hypot(*np.diff(seg, axis=0).T).sum()) / s
        rac = []
        for end, i_out in (('debut', a - 1), ('fin', b)):
            if i_out < 0 or i_out >= len(d):
                continue  # end of the piece itself: no partner track here
            k = int(np.argmin(dist[i_out]))
            g = geoms[k]
            tip = d[i_out]
            proj = g.interpolate(g.project(shapely.Point(tip)))
            # heading of the branch over the last CAP_M metres, oriented towards the cut end
            cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(seg, axis=0).T))]) / s
            if end == 'fin':
                j0 = int(np.searchsorted(cum, cum[-1] - CAP_M))
                hb = heading(seg[min(j0, len(seg) - 2)], seg[-1])
            else:
                j0 = int(np.searchsorted(cum, CAP_M))
                hb = heading(seg[min(j0, len(seg) - 1)], seg[0])
            rac.append({'bout': end, 'raccord_merc': [proj.x, proj.y], 'raccord_ecart_m': round(float(dmin[i_out]), 1),
                        'cap_branche_deg': round(hb, 1), 'cap_voie_deg': round(axis_heading(g, (proj.x, proj.y), hb, s), 1),
                        'partner_line_id': partners[k]['line_id'], 'partner_source': partners[k]['src'],
                        'jonction_dist_m': round(min(proj.distance(p) for p in jp) / s, 1) if jp else None})
        if L < (min_raccord_m if rac else min_free_m):
            continue
        out.append((np.asarray(LineString(seg).simplify(0.3 * s).coords), rac))
    return out


def merge_with_existing(new_merc: list[np.ndarray], existing_merc: list[np.ndarray], s: float,
                        overlap_m=8.0, replace_frac=0.6, min_left_m=MIN_RACCORD_M):
    """Reconciles the junction pieces with the existing national pieces (same line, same layer).

    - an existing piece covered at >= replace_frac (within overlap_m) by the new ones is REPLACED;
    - otherwise it is kept, and the part of the new ones that duplicates it (within overlap_m) is removed.
    Returns (indices of the existing pieces to delete, list per new piece of the remaining pieces).
    """
    if not existing_merc:
        return [], [[np.asarray(n)] for n in new_merc]
    N = MultiLineString([LineString(n) for n in new_merc]) if new_merc else None
    drop, keep = [], []
    for i, e in enumerate(existing_merc):
        el = LineString(e)
        sp = shapely.points(_densify(np.asarray(e, float), 5.0 * s))
        frac = float((shapely.distance(sp, N) / s <= overlap_m).mean()) if N is not None else 0.0
        (drop if frac >= replace_frac else keep).append(i)
    rest = []
    K = shapely.union_all([LineString(existing_merc[i]).buffer(overlap_m * s) for i in keep]) if keep else None
    for n in new_merc:
        nl = LineString(n)
        if K is None:
            rest.append([np.asarray(n)])
            continue
        diff = nl.difference(K)
        parts = [np.asarray(g.coords) for g in shapely.get_parts(shapely.line_merge(diff) if not diff.is_empty else diff)
                 if g.geom_type == 'LineString' and g.length / s >= min_left_m]
        rest.append(parts)
    return drop, rest
