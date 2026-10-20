"""Inference in the corridor of the coarse traces, vectorisation and registration.

For each target portion (coarse trace without precise data, or,
in "eval" mode, coarse trace of a line whose precise axis is known):

1. Corridor around the coarse trace: base half-width per source
   (config.coarse_sources: 120 m by default, 200 m SNCF) + 15 % of the longest local chord, capped at 400 m.
2. Corridor tiles (+100 m) downloaded politely, "track" probability per
   tile (3 × 3 tile window for context, TTA 4 rotations), cached.
3. Minimum-cost path (MCP) in the corridor, split into overlapping 4 km
   chunks (start / end = transverse sections, no imposed point).
   Cost = -log p + 0.15 + soft penalty for distance from the coarse trace;
   the precise tracks of OTHER lines are forbidden (no track "jumping").
4. Only the parts of the path SUPPORTED by the model (smoothed p >= 0.35 over
   >= 200 m, no bend with radius < 60 m) are kept: nothing is invented where
   the model sees nothing (the coarse trace remains the reference there).
5. Registration of the basemap georeferencing: offset field estimated on the neighbouring
   precise tracks (field.OffsetField), subtracted from the extracted pieces.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import shapely
import shapely.ops
import torch
from PIL import Image, ImageDraw
from shapely import LineString, STRtree, box

from . import config
from . import db as vdb
from .dataset import chip_fold, fetch_truth
from .field import OffsetField, normal_profiles
from .geo import TILE, lonlat_to_merc, merc_scale, merc_to_lonlat, merc_to_px, px_to_merc
from .tiles import LAYERS, PoliteFetcher, TileCache

BASE_W = config.coarse_sources()   # coarse trace source -> base half-width of the corridor (m)
COARSE = tuple(BASE_W)
CHUNK_M = 4000.0
EXT_M = 600.0
SUPPORT_P = 0.35
MIN_RUN_M = 200.0
MIN_RADIUS_M = 60.0


# ===== targets =====================================================================

def fetch_targets(conn, args, truth_line_ids):
    env = vdb.env_sql(args.bbox)
    lines = ''
    if args.lines:
        lines = 'AND p.line_id = ANY(%(lines)s)'
    if getattr(args, 'exclude_networks', None):
        lines += ' AND NOT EXISTS (SELECT 1 FROM lines l WHERE l.id = p.line_id AND l.network = ANY(%(excl)s))'
    if args.targets == 'file':
        return fetch_file_targets(args)
    params = {'excl': (args.exclude_networks or '').split(','), 'lines': [int(v) for v in args.lines.split(',')] if args.lines else None, 'truth': list(truth_line_ids), 'scope': getattr(args, 'eval_scope', 'truth'), 'coarse': list(COARSE)}
    if args.targets == 'coarse':
        rows = conn.execute(f"""
            SELECT p.id, p.line_id, p.source, ST_AsBinary(ST_Transform(p.geom, 3857)), NULL
            FROM line_reference_parts p
            WHERE p.matched_source IS NULL AND p.source = ANY(%(coarse)s)
              AND ST_Intersects(ST_Centroid(p.geom), {env}) {lines}
            ORDER BY p.line_id, p.id
        """, params).fetchall()
    else:
        # Original coarse trace under a piece registered on a precise track of a ground-truth line.
        rows = conn.execute(f"""
            SELECT p.id, p.line_id, p.source,
                   ST_AsBinary(ST_Transform(ST_LineMerge(ST_CollectionExtract(ST_Intersection(
                       ST_Transform(t.g, 2154), ST_Buffer(ST_Transform(p.geom, 2154), 150)), 2)), 3857)),
                   ST_AsBinary(ST_Transform(p.geom, 3857))
            FROM line_reference_parts p
            CROSS JOIN LATERAL (SELECT ST_Collect(geom) AS g FROM source_traces
                                WHERE id IN (SELECT jsonb_array_elements_text(p.source_trace_ids)::bigint)) t
            WHERE p.matched_source IN ('bdtopo', 'osm') AND p.source = ANY(%(coarse)s)
              AND p.length_m >= 1000 AND ((%(scope)s = 'truth' AND p.line_id = ANY(%(truth)s)) OR %(scope)s = 'all' OR (%(scope)s = 'closed' AND NOT p.line_id = ANY(%(truth)s)))
              AND ST_Intersects(ST_Centroid(p.geom), {env}) {lines}
            ORDER BY p.line_id, p.id
        """, params).fetchall()
    out = []
    for r in rows:
        g = vdb.geoms([r], 3)[0]
        truth = vdb.geoms([r], 4)[0] if r[4] is not None else None
        for k, ls in enumerate(vdb.lines_of(g)):
            line = LineString(ls)
            if line.length < 150 * merc_scale(48):
                continue
            out.append({'tid': f'{r[0]}-{k}', 'part_id': int(r[0]), 'line_id': int(r[1]), 'source': r[2],
                        'coarse': ls, 'truth': [a.tolist() for a in vdb.lines_of(truth)] if truth is not None else None})
    deg = getattr(args, 'degrade_m', 0) or 0
    if deg and args.targets == 'eval':
        # Realistic evaluation: simplified coarse trace (chords) like the portions without precise data.
        for t in out:
            c = np.asarray(t['coarse'])
            lat = float(merc_to_lonlat(c[:, 0].mean(), c[:, 1].mean())[1])
            t['coarse'] = np.asarray(LineString(c).simplify(deg * merc_scale(lat)).coords)
    if args.limit:
        out = out[:args.limit]
    return out


def fetch_file_targets(args):
    """Prepared targets (`rail.py junctions`): junction corridors, selected by centroid within the bbox."""
    lon0, lat0, lon1, lat1 = args.bbox
    lines = {int(v) for v in args.lines.split(',')} if args.lines else None
    out = []
    for row in open(args.targets_file):
        t = json.loads(row)
        c = np.asarray(t['coarse'])
        lon, lat = merc_to_lonlat(c[:, 0].mean(), c[:, 1].mean())
        if not (lon0 <= float(lon) < lon1 and lat0 <= float(lat) < lat1):
            continue
        if lines is not None and t['line_id'] not in lines:
            continue
        if LineString(c).length < 150 * merc_scale(float(lat)):
            continue
        t['coarse'] = c
        out.append(t)
    if args.limit:
        out = out[:args.limit]
    return out


def is_eval(args, targets) -> bool:
    return args.targets == 'eval' or (args.targets == 'file' and any(t.get('truth') for t in targets))


def prob_root_for(work: Path, model: str, layer_key: str) -> Path:
    """Probability cache: per model AND per basemap (ortho-1950 keeps the old path)."""
    return work / 'prob' / (model if layer_key == 'ortho-1950' else f'{model}@{layer_key}')


def fetch_blockers(conn, bbox, margin_deg=0.05):
    """Precise tracks (all lines): forbidden to the path, except the targeted line itself."""
    lon0, lat0, lon1, lat1 = bbox
    env = vdb.env_sql((lon0 - margin_deg, lat0 - margin_deg, lon1 + margin_deg, lat1 + margin_deg))
    rows = conn.execute(f"""
        SELECT COALESCE(line_id, 0), ST_AsBinary(ST_Transform(geom, 3857)) FROM source_traces
        WHERE geom && {env} AND (
            (source = 'bdtopo' AND layer IN ('LGV', 'Voie ferrée principale', 'Sans objet', 'Voie de service'))
            OR (source = 'osm' AND layer NOT IN ('subway', 'tram', 'monorail', 'funicular', 'miniature', 'proposed', 'construction')))
    """).fetchall()
    out = []
    for (lid, _), g in zip(rows, vdb.geoms(rows, 1)):
        for ls in vdb.lines_of(g):
            out.append((int(lid), ls))
    return out


def corridor_width(coarse_merc, source, lat):
    """Half-width (ground m): base + 15 % of the longest chord, capped."""
    seg = np.diff(coarse_merc, axis=0)
    chord = float(np.hypot(seg[:, 0], seg[:, 1]).max()) / merc_scale(lat)
    return float(np.clip(BASE_W.get(source, 150.0) + 0.15 * chord, BASE_W.get(source, 150.0), 400.0))


def chunks_of(target, z):
    """Splits a target into 4 km chunks (± 600 m overlap), in global px."""
    c = np.asarray(target['coarse'])
    lon, lat = merc_to_lonlat(c[:, 0].mean(), c[:, 1].mean())
    lat = float(lat)
    px, py = merc_to_px(c[:, 0], c[:, 1], z)
    line = LineString(np.stack([px, py], 1))
    gres = (2 * 20037508.342789244 / (TILE * 2 ** z)) * math.cos(math.radians(lat))
    L = line.length
    step, ext = CHUNK_M / gres, EXT_M / gres
    out = []
    a = 0.0
    while a < L - 1e-6:
        b = min(L, a + step)
        if L - b < 0.3 * step:
            b = L
        a_ext, b_ext = max(0.0, a - ext), min(L, b + ext)
        sub = shapely.ops.substring(line, a_ext, b_ext)
        chords = np.diff(np.asarray(sub.coords), axis=0)
        chord_m = float(np.hypot(chords[:, 0], chords[:, 1]).max()) * gres
        W = float(np.clip(BASE_W.get(target['source'], 150.0) + 0.15 * chord_m, BASE_W.get(target['source'], 150.0), 400.0))
        out.append({'a': a, 'b': b, 'a_ext': a_ext, 'b_ext': b_ext, 'W_m': W, 'gres': gres, 'lat': lat})
        a = b
    return line, out


def tiles_for_targets(targets, z, margin_m=100.0):
    tiles = set()
    for t in targets:
        line, chs = chunks_of(t, z)
        t['_chunks'] = chs
        for ch in chs:
            sub = shapely.ops.substring(line, ch['a_ext'], ch['b_ext'])
            poly = sub.buffer((ch['W_m'] + margin_m) / ch['gres'], cap_style='flat')
            x0, y0, x1, y1 = poly.bounds
            for tx in range(int(x0 // TILE), int(x1 // TILE) + 1):
                for ty in range(int(y0 // TILE), int(y1 // TILE) + 1):
                    if poly.intersects(box(tx * TILE, ty * TILE, (tx + 1) * TILE, (ty + 1) * TILE)):
                        tiles.add((tx, ty))
    return tiles


# ===== GPU =========================================================================

class WindowDataset(torch.utils.data.Dataset):
    """3 × 3 tile window centred on each tile to predict (missing tiles: median colour)."""

    def __init__(self, cache: TileCache, z, tiles):
        self.cache, self.z, self.tiles = cache, z, tiles

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        x, y = self.tiles[i]
        win = np.zeros((3 * TILE, 3 * TILE, 3), np.uint8)
        center = self.cache.read(self.z, x, y)
        if center is None:
            return torch.zeros((3 * TILE, 3 * TILE, 3), dtype=torch.uint8), x, y, False
        fill = np.median(center.reshape(-1, 3), axis=0).astype(np.uint8)
        for j in range(3):
            for k in range(3):
                t = center if (j, k) == (1, 1) else self.cache.read(self.z, x + k - 1, y + j - 1)
                win[j * TILE:(j + 1) * TILE, k * TILE:(k + 1) * TILE] = t if t is not None and t.shape == (TILE, TILE, 3) else fill
        return torch.from_numpy(win), x, y, True


def prob_path(root: Path, z, x, y):
    return root / str(z) / str(x) / f'{y}.png'


def predict_tiles(model, device, cache, z, tiles, prob_root: Path, batch=12, workers=4, tta=4, log=print, max_util=80.0):
    from .throttle import GpuThrottle
    from .train import MEAN, STD

    throttle = GpuThrottle(max_util, log=log)

    todo = [t for t in tiles if not prob_path(prob_root, z, *t).exists() and cache.path(z, *t).exists()]
    log(f'  probabilities: {len(tiles)} tiles, {len(todo)} to predict')
    if not todo:
        return
    dl = torch.utils.data.DataLoader(WindowDataset(cache, z, todo), batch_size=batch, num_workers=workers,
                                     pin_memory=True, prefetch_factor=4 if workers else None)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    saver = ThreadPoolExecutor(4)
    t0, n = time.time(), 0

    def save(p, arr):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(p, compress_level=1)

    with torch.no_grad():
        for win, xs, ys, ok in dl:
            throttle.start()
            xb = win.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
            xb = ((xb - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                acc = None
                rots = {1: [0], 2: [0, 2], 4: [0, 1, 2, 3]}[int(tta) if tta else 1]
                for k in rots:
                    p = torch.sigmoid(model(torch.rot90(xb, k, (2, 3))).float())
                    p = torch.rot90(p, -k, (2, 3))
                    acc = p if acc is None else acc + p
                prob = (acc / len(rots))[:, 0, TILE:2 * TILE, TILE:2 * TILE]
            arr = (prob.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            throttle.end()
            for b in range(arr.shape[0]):
                if bool(ok[b]):
                    saver.submit(save, prob_path(prob_root, z, int(xs[b]), int(ys[b])), arr[b])
            n += arr.shape[0]
            if n % (batch * 100) < batch:
                log(f'  {n}/{len(todo)} tiles predicted ({n / (time.time() - t0):.1f} t/s)')
    saver.shutdown(wait=True)
    log(f'  {n} tiles predicted in {time.time() - t0:.0f} s; {throttle.summary()}')


def read_prob(prob_root, z, x, y):
    p = prob_path(prob_root, z, x, y)
    if not p.exists():
        return None
    return np.asarray(Image.open(p), np.float32) / 255.0


def observations(prob_root, z, tiles, truth, search_m=60.0):
    """Offset observations on the precise in-service tracks (truth) crossing the tiles."""
    tree = STRtree([LineString(ls) for _, ls in truth])
    obs = []
    for (x, y) in tiles:
        x0, y0 = x * TILE, y * TILE
        mx0, my0 = px_to_merc(x0, y0, z)
        mx1, my1 = px_to_merc(x0 + TILE, y0 + TILE, z)
        idx = tree.query(box(mx0, my1, mx1, my0))
        if not len(idx):
            continue
        prob = read_prob(prob_root, z, x, y)
        if prob is None:
            continue
        _, lat = merc_to_lonlat((mx0 + mx1) / 2, (my0 + my1) / 2)
        gres = (2 * 20037508.342789244 / (TILE * 2 ** z)) * math.cos(math.radians(float(lat)))
        for i in idx:
            line_id, ls = truth[i]
            px, py = merc_to_px(ls[:, 0], ls[:, 1], z)
            o = normal_profiles(prob, [np.stack([px - x0, py - y0], 1)], search_m / gres)
            if not len(o):
                continue
            mx, my = px_to_merc(o[:, 0] + x0, o[:, 1] + y0, z)
            for k in range(len(o)):
                obs.append((mx[k], my[k], o[k, 2], -o[k, 3], o[k, 4] * gres, o[k, 5], line_id, 0, float(lat)))
    return np.asarray(obs, float).reshape(-1, 9)


# ===== path ========================================================================

_W = {}


def _init_worker(prob_root, z, blockers, field_obs, eval_mode, apply_field=True):
    from skimage.graph import MCP_Geometric  # noqa: F401 (early import in the process)

    _W['prob_root'], _W['z'], _W['eval'] = Path(prob_root), z, eval_mode
    _W['blockers'] = blockers
    _W['btree'] = STRtree([LineString(ls) for _, ls in blockers]) if blockers else None
    fo = field_obs
    _W['field'] = OffsetField(fo[:, 0], fo[:, 1], fo[:, 2], fo[:, 3], fo[:, 4], w=fo[:, 5], group=fo[:, 6]) if len(fo) and apply_field else None


def _mosaic(x0, y0, x1, y1):
    z, root = _W['z'], _W['prob_root']
    tx0, ty0, tx1, ty1 = int(x0 // TILE), int(y0 // TILE), int(x1 // TILE), int(y1 // TILE)
    P = np.zeros(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE), np.float32)
    have = np.zeros_like(P, dtype=bool)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            p = read_prob(root, z, tx, ty)
            if p is not None:
                P[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = p
                have[(ty - ty0) * TILE:(ty - ty0 + 1) * TILE, (tx - tx0) * TILE:(tx - tx0 + 1) * TILE] = True
    return P, have, tx0 * TILE, ty0 * TILE


def _radius_ok(pts_m, win=20.0):
    """Mask of the points where the local curvature radius (window ± win m) stays >= MIN_RADIUS_M."""
    seg = np.diff(pts_m, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(L)])
    if cum[-1] < 3 * win:
        return np.ones(len(pts_m), bool)
    s = np.arange(0, cum[-1], 10.0)
    x = np.interp(s, cum, pts_m[:, 0])
    y = np.interp(s, cum, pts_m[:, 1])
    h = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    k = int(win / 10)
    dh = np.abs(h[2 * k:] - h[:-2 * k]) if len(h) > 2 * k else np.zeros(0)
    radius = np.full(len(s), np.inf)
    if len(dh):
        radius[k:-k] = (2 * win) / np.maximum(dh, 1e-6)
    ok_s = radius >= MIN_RADIUS_M
    return np.interp(cum, s, ok_s.astype(float)) > 0.5


def _process_target(t):
    from scipy import ndimage
    from skimage.graph import MCP_Geometric

    z = _W['z']
    c = np.asarray(t['coarse'])
    line, chunks = chunks_of(t, z)
    runs_out, stats = [], {'tid': t['tid'], 'line_id': t['line_id'], 'part_id': t['part_id'], 'source': t['source'],
                           'len_m': 0.0, 'kept_m': 0.0, 'chunks': len(chunks), 'no_tiles': 0}
    gres0 = chunks[0]['gres']
    stats['len_m'] = line.length * gres0
    own_line = t['line_id']
    truth_geom = None
    if _W['eval'] and t['truth']:
        tl = []
        for ls in t['truth']:
            a = np.asarray(ls)
            px, py = merc_to_px(a[:, 0], a[:, 1], z)
            tl.append(LineString(np.stack([px, py], 1)))
        truth_geom = shapely.MultiLineString(tl)
    for ch in chunks:
        gres, W = ch['gres'], ch['W_m']
        sub = shapely.ops.substring(line, ch['a_ext'], ch['b_ext'])
        if sub.length < 5:
            continue
        Wp = W / gres
        poly = sub.buffer(Wp, cap_style='flat')
        x0, y0, x1, y1 = poly.bounds
        P, have, ox, oy = _mosaic(x0 - 2, y0 - 2, x1 + 2, y1 + 2)
        if have.mean() < 0.05:
            stats['no_tiles'] += 1
            continue
        H, Wd = P.shape
        corr = Image.new('L', (Wd, H), 0)
        dr = ImageDraw.Draw(corr)
        geoms = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
        for g in geoms:
            dr.polygon([(px - ox, py - oy) for px, py in g.exterior.coords], fill=1)
            for hole in g.interiors:
                dr.polygon([(px - ox, py - oy) for px, py in hole.coords], fill=0)
        corr = np.asarray(corr, bool) & have
        # Precise tracks of other lines: forbidden (± 12 m), except near the ends of the target.
        block = Image.new('L', (Wd, H), 0)
        if _W['btree'] is not None:
            mx0, my0 = px_to_merc(ox, oy, z)
            mx1, my1 = px_to_merc(ox + Wd, oy + H, z)
            bd = ImageDraw.Draw(block)
            for i in _W['btree'].query(box(mx0, my1, mx1, my0)):
                lid, ls = _W['blockers'][i]
                if lid == own_line:
                    continue
                bpx, bpy = merc_to_px(ls[:, 0], ls[:, 1], z)
                pts = np.stack([bpx, bpy], 1)
                if truth_geom is not None:
                    # evaluation mode: do not block the tracks merged with the evaluated axis
                    d = shapely.distance(shapely.points(pts), truth_geom)
                    if np.median(d) * gres < 20:
                        continue
                bd.line([(a - ox, b - oy) for a, b in pts], fill=1, width=max(1, int(round(24 / gres))))
        block = np.asarray(block, bool)
        if t.get('release') is not None:  # junction corridor: released only at the junction points
            rel = [shapely.Point(*merc_to_px(p[0], p[1], z)) for p in t['release']]
        else:  # ends of the portion: connection to the neighbouring precise track
            rel = [line.interpolate(0), line.interpolate(line.length)]
        for e in rel:
            ex, ey = int(e.x - ox), int(e.y - oy)
            r = int(60 / gres)
            block[max(0, ey - r):ey + r, max(0, ex - r):ex + r] = False
        passable = corr & ~block
        # Soft penalty for distance from the coarse trace.
        dist_img = Image.new('L', (Wd, H), 0)
        ImageDraw.Draw(dist_img).line([(a - ox, b - oy) for a, b in sub.coords], fill=1, width=1)
        dcoarse = ndimage.distance_transform_edt(~np.asarray(dist_img, bool)) / Wp
        cost = -np.log(np.clip(P, 0.03, 1.0)) + 0.15 + 0.6 * dcoarse ** 2
        cost[~passable] = np.inf
        # Start / end sections: "flat" edges of the corridor.
        sc = np.asarray(sub.coords)

        def cap(pt, nxt):
            d = np.asarray(nxt) - np.asarray(pt)
            d = d / max(np.hypot(*d), 1e-6)
            n = np.array([-d[1], d[0]])
            seg = [tuple(np.asarray(pt) + n * Wp - (ox, oy)), tuple(np.asarray(pt) - n * Wp - (ox, oy))]
            im = Image.new('L', (Wd, H), 0)
            ImageDraw.Draw(im).line(seg, fill=1, width=3)
            m = np.asarray(im, bool) & passable
            return np.argwhere(m)

        starts = cap(sc[0], sc[1])
        endpts = cap(sc[-1], sc[-2])
        if not len(starts) or not len(endpts):
            continue
        mcp = MCP_Geometric(cost, fully_connected=True)
        cum, _ = mcp.find_costs([tuple(s) for s in starts])
        ce = cum[endpts[:, 0], endpts[:, 1]]
        if not np.isfinite(ce).any():
            continue
        end = tuple(endpts[int(np.argmin(ce))])
        path = np.asarray(mcp.traceback(end), float)  # (row, col)
        pp = P[path[:, 0].astype(int), path[:, 1].astype(int)]
        # Keep only the central part of the chunk [a, b].
        gpts = np.stack([path[:, 1] + ox, path[:, 0] + oy], 1)
        along = shapely.line_locate_point(line, shapely.points(gpts))
        keep = (along >= ch['a']) & (along <= ch['b'])
        # Support: probability smoothed over ~60 m.
        k = max(3, int(60 / gres) | 1)
        ps = ndimage.uniform_filter1d(pp, size=k, mode='nearest')
        sup = (ps >= SUPPORT_P) & keep
        # Supported runs.
        idx = np.flatnonzero(np.diff(np.concatenate([[0], sup.astype(int), [0]])))
        for s0, s1 in zip(idx[::2], idx[1::2]):
            seg = gpts[s0:s1]
            if len(seg) < 3:
                continue
            seg_len = np.hypot(*np.diff(seg, axis=0).T).sum() * gres
            if seg_len < MIN_RUN_M:
                continue
            # Smoothing (moving average ~15 m) then curvature check.
            kk = max(3, int(15 / gres) | 1)
            sm = np.stack([ndimage.uniform_filter1d(seg[:, i], size=kk, mode='nearest') for i in range(2)], 1)
            okr = _radius_ok(sm * gres)
            # split at bends (± 40 m removed)
            bad = ~okr
            if bad.any():
                badd = ndimage.binary_dilation(bad, iterations=max(1, int(40 / gres)))
                pieces = np.flatnonzero(np.diff(np.concatenate([[0], (~badd).astype(int), [0]])))
                segs = [(sm[a:b], pp[s0 + a:s0 + b]) for a, b in zip(pieces[::2], pieces[1::2])]
            else:
                segs = [(sm, pp[s0:s1])]
            for spx, sp in segs:
                if len(spx) < 3:
                    continue
                L = np.hypot(*np.diff(spx, axis=0).T).sum() * gres
                if L < MIN_RUN_M:
                    continue
                ls = LineString(spx).simplify(0.5)
                mx, my = px_to_merc(np.asarray(ls.coords)[:, 0], np.asarray(ls.coords)[:, 1], z)
                dco = shapely.distance(shapely.points(spx), sub) * gres
                runs_out.append({'merc': np.stack([mx, my], 1), 'len_m': float(L), 'mean_p': float(sp.mean()),
                                 'frac_p50': float((sp >= 0.5).mean()), 'ecart_grossier_med_m': float(np.median(dco)),
                                 'ecart_grossier_max_m': float(dco.max()), 'corridor_m': W, 'gres': gres,
                                 'lat': ch['lat']})
    # Georeferencing registration and confidence.
    field = _W['field']
    feats = []
    for r in runs_out:
        m = r['merc']
        cx, cy = m[len(m) // 2]
        f = field.at(cx, cy, merc_scale(r['lat']), exclude_group=own_line if _W['eval'] else None) if field else None
        if f is not None:
            dx, dy, nobs, sd = f
            s = merc_scale(r['lat'])
            m = m - np.array([dx * s, dy * s])
        else:
            dx = dy = sd = None
            nobs = 0
        conf = r['mean_p'] * (0.5 + 0.5 * r['frac_p50']) * (1.0 if (f is not None or field is None) else 0.85)
        lon, lat = merc_to_lonlat(m[:, 0], m[:, 1])
        feats.append({'coords': np.round(np.stack([lon, lat], 1), 7).tolist(), 'merc': m.tolist(),
                      'props': {'len_m': round(r['len_m'], 1), 'mean_p': round(r['mean_p'], 3), 'frac_p50': round(r['frac_p50'], 3),
                                'confidence': round(conf, 3), 'ecart_grossier_med_m': round(r['ecart_grossier_med_m'], 1),
                                'ecart_grossier_max_m': round(r['ecart_grossier_max_m'], 1), 'corridor_m': round(r['corridor_m']),
                                'recalage_dx_m': None if dx is None else round(dx, 1), 'recalage_dy_m': None if dy is None else round(dy, 1),
                                'recalage_obs': nobs, 'recalage_sd_m': None if sd is None else round(sd, 1)}})
        stats['kept_m'] += r['len_m']
    return stats, feats


# ===== commands ====================================================================

def run(args, log=print):
    from .train import load_model

    layer = LAYERS[args.layer]
    z = args.zoom or layer.zoom
    work = Path(args.work)
    out = work / 'runs' / args.run
    out.mkdir(parents=True, exist_ok=True)
    conn = vdb.connect()
    years = [int(y) for y in args.eras.split(',')]
    if args.targets == 'file':
        # prepared targets: the useful extent (forbidden tracks, truth) is reduced to that of the targets
        targets = fetch_file_targets(args)
        if targets:
            allc = np.concatenate([np.asarray(t['coarse']) for t in targets])
            (lo0, lo1), (la0, la1) = (merc_to_lonlat(allc[:, 0].min(), allc[:, 1].min()), merc_to_lonlat(allc[:, 0].max(), allc[:, 1].max()))
            args.bbox = (float(lo0) - 0.01, float(la0) - 0.01, float(lo1) + 0.01, float(la1) + 0.01)
        truth = fetch_truth(conn, args.bbox, years, margin_deg=0.1)
    else:
        truth = fetch_truth(conn, args.bbox, years, margin_deg=0.1)
        targets = fetch_targets(conn, args, sorted({lid for lid, _ in truth}))
    log(f'{len(targets)} target portions ({args.targets}), {sum(LineString(t["coarse"]).length for t in targets) / merc_scale(47) / 1000:.0f} km '
        f'; ground truth {len(truth)} axes ({len({lid for lid, _ in truth})} lines)')
    tiles = sorted(tiles_for_targets(targets, z))
    cache = TileCache(args.tiles, layer)
    log(f'{len(tiles)} tiles in the corridors')
    if not args.no_download:
        PoliteFetcher(cache, workers=args.dl_workers, max_rps=args.max_rps, log=log).fetch_all(tiles, z)
    if getattr(args, 'download_only', False):
        return

    from .throttle import limit_memory

    device = torch.device('cuda' if torch.cuda.is_available() and not getattr(args, 'no_predict', False) else 'cpu')
    if device.type == 'cuda':
        limit_memory(args.gpu_max_mem_gb)
    model, ck = load_model(work / 'models' / args.model / 'best.pt', device)
    prob_root = prob_root_for(work, args.model, layer.key)
    if getattr(args, 'no_predict', False):
        miss = sum(1 for t in tiles if not prob_path(prob_root, z, *t).exists())
        log(f'  --no-predict: {miss}/{len(tiles)} tiles without probability (ignored)')
    else:
        predict_tiles(model, device, cache, z, tiles, prob_root, batch=args.batch, workers=args.workers, log=log,
                      max_util=args.gpu_max_util, tta=args.tta)
    del model
    torch.cuda.empty_cache()

    if layer.recalage:
        obs = observations(prob_root, z, tiles, truth)
        extra = work / 'models' / args.model
        for f in extra.glob('offsets-obs-*.npy'):
            o = np.load(f)
            obs = np.concatenate([obs, o[:, :9]]) if len(obs) else o[:, :9]
    else:  # orthophotos: no registration field (it degrades), no need to measure it
        obs = np.zeros((0, 9))
    log(f'{len(obs)} registration observations')
    np.save(out / 'obs.npy', obs)
    blockers = fetch_blockers(conn, args.bbox)
    log(f'{len(blockers)} precise tracks (forbidden outside the targeted line)')

    for t in targets:
        t.pop('_chunks', None)
    ctx = mp.get_context('fork')
    feats, stats = [], []
    t0 = time.time()
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(prob_root, z, blockers, obs, is_eval(args, targets), layer.recalage)) as pool:
        for i, (st, fs) in enumerate(pool.imap_unordered(_process_target, targets, chunksize=1)):
            stats.append(st)
            for f in fs:
                f['props'].update({'line_id': st['line_id'], 'part_id': st['part_id'], 'tid': st['tid'], 'coarse_source': st['source']})
                feats.append(f)
            if (i + 1) % 100 == 0:
                log(f'  {i + 1}/{len(targets)} portions ({time.time() - t0:.0f} s)')
    tot = sum(s['len_m'] for s in stats)
    kept = sum(s['kept_m'] for s in stats)
    summary = {'layer': layer.key, 'wmts': layer.layer, 'attribution': layer.label, 'zoom': z, 'model': args.model,
               'model_metrics': ck.get('metrics'),
               'targets': args.targets if args.targets != 'file' else ('jonction-eval' if is_eval(args, targets) else 'jonction'), 'bbox': args.bbox, 'portions': len(targets),
               'km_cibles': round(tot / 1000, 1), 'km_extraits': round(kept / 1000, 1), 'tuiles': len(tiles),
               'observations_recalage': int(len(obs))}
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1, ensure_ascii=False)
    with open(out / 'targets.jsonl', 'w') as f:
        for t, s in zip(sorted(targets, key=lambda t: t['tid']), sorted(stats, key=lambda s: s['tid'])):
            extra = {k: t[k] for k in ('junctions', 'precise', 'release') if k in t}
            f.write(json.dumps({**s, 'coarse': t['coarse'].tolist() if hasattr(t['coarse'], 'tolist') else t['coarse'],
                                'truth': t['truth'], **extra}) + '\n')
    gj = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': f['coords']}, 'properties': f['props']} for f in feats]}
    json.dump(gj, open(out / 'extracted.geojson', 'w'))
    log(json.dumps(summary, ensure_ascii=False))
    log(f'done: {len(feats)} pieces extracted -> {out / "extracted.geojson"}')


def _sample(line: LineString, step):
    n = max(2, int(line.length / step) + 1)
    return shapely.line_interpolate_point(line, np.linspace(0, line.length, n))


def evaluate(args, log=print):
    """Compares the extracted pieces (eval mode) to the known precise axis, and to the coarse trace."""
    out = Path(args.work) / 'runs' / args.run
    summary = json.load(open(out / 'summary.json'))
    z = summary['zoom']
    feats = json.load(open(out / 'extracted.geojson'))['features']
    by_tid = {}
    for f in feats:
        by_tid.setdefault(f['properties']['tid'], []).append(f)
    res = {k: {'ext': [], 'tru_cov': [], 'coarse': [], 'final': [], 'L_ext': 0, 'L_tru': 0, 'ext_f': [], 'L_ext_f': 0, 'L_raw': 0}
           for k in ('val', 'train')}
    for line in open(out / 'targets.jsonl'):
        t = json.loads(line)
        if not t['truth']:
            continue
        c = np.asarray(t['coarse'])
        lat = float(merc_to_lonlat(c[:, 0].mean(), c[:, 1].mean())[1])
        s = merc_scale(lat)
        truth = shapely.MultiLineString([LineString(a) for a in t['truth']])
        coarse = LineString(c)
        # fold (validation / training) of the block containing the middle of the portion
        mid = coarse.interpolate(0.5, normalized=True)
        px, py = merc_to_px(mid.x, mid.y, z)
        fold = chip_fold(z, int(px // 512), int(py // 512))
        R = res[fold]
        ext_m = []
        for f in by_tid.get(t['tid'], []):
            ll = np.asarray(f['geometry']['coordinates'])
            mx, my = lonlat_to_merc(ll[:, 0], ll[:, 1])
            ext_m.append(LineString(np.stack([mx, my], 1)))
        # offset of the extracted pieces from the truth (ground m); "_f" = after the writing filter
        for f, e in zip(by_tid.get(t['tid'], []), ext_m):
            d = shapely.distance(_sample(e, 10 * s), truth) / s
            R['ext'].extend(d.tolist())
            R['L_ext'] += e.length / s
            p = f['properties']
            keep = p['confidence'] >= 0.7
            if keep:
                keep, _, _ = vertex_check(f['geometry']['coordinates'], t['coarse'], t['source'], p)
                keep = keep and (t['source'] == 'sncf' or sag_check(f['geometry']['coordinates'], t['coarse']) <= 0.05)
            if keep:
                R['ext_f'].extend(d.tolist())
                R['L_ext_f'] += e.length / s
        # coverage of the truth by the extraction, and offset of the coarse trace
        tp = _sample(coarse, 10 * s)
        dc = shapely.distance(tp, truth) / s
        R['coarse'].extend(dc.tolist())
        tsamp = np.concatenate([np.asarray([(p.x, p.y) for p in _sample(l, 10 * s)]) for l in truth.geoms])
        tpts = shapely.points(tsamp)
        # only the truth "facing" the coarse trace counts (the registered piece may overflow)
        near = shapely.distance(tpts, coarse) / s < 200
        tpts = tpts[near]
        R['L_tru'] += len(tpts) * 10
        if ext_m:
            E = shapely.MultiLineString(ext_m)
            de = shapely.distance(tpts, E) / s
            R['tru_cov'].extend(de.tolist())
            # final trace: the extraction replaces the coarse one over the extent it covers along the
            # coarse trace (projection of its ends), without looking at the truth.
            st = shapely.line_locate_point(coarse, tpts)
            cov = np.zeros(len(tpts), bool)
            for e in ext_m:
                a, b = sorted(shapely.line_locate_point(coarse, shapely.points(np.asarray(e.coords)[[0, -1]])))
                cov |= (st >= a) & (st <= b)
            df = np.where(cov, de, shapely.distance(tpts, coarse) / s)
        else:
            R['tru_cov'].extend([np.inf] * len(tpts))
            df = shapely.distance(tpts, coarse) / s
        R['final'].extend(df.tolist())

    def pct(a, q):
        return round(float(np.percentile(a, q)), 1) if len(a) else None

    report = {}
    for fold, R in res.items():
        e, cov, co, fi = (np.asarray(R[k]) for k in ('ext', 'tru_cov', 'coarse', 'final'))
        ef = np.asarray(R['ext_f'])
        p10 = float((e <= 10).mean()) if len(e) else 0.0
        r10 = float((cov <= 10).mean()) if len(cov) else 0.0
        report[fold] = {
            'f1_10m': round(2 * p10 * r10 / (p10 + r10), 3) if p10 + r10 else None,
            'filtre_km_gardes': round(R['L_ext_f'] / 1000, 1),
            'filtre_faux_sup_20m_pct': round(100 * float((ef > 20).mean()), 1) if len(ef) else None,
            'filtre_ecart_median_m': pct(ef, 50), 'filtre_ecart_p90_m': pct(ef, 90),
            'sans_filtre_faux_sup_20m_pct': round(100 * float((e > 20).mean()), 1) if len(e) else None,
            'km_extraits': round(R['L_ext'] / 1000, 1), 'km_verite': round(R['L_tru'] / 1000, 1),
            'precision_10m': round(float((e <= 10).mean()), 3) if len(e) else None,
            'precision_20m': round(float((e <= 20).mean()), 3) if len(e) else None,
            'rappel_10m': round(float((cov <= 10).mean()), 3) if len(cov) else None,
            'rappel_20m': round(float((cov <= 20).mean()), 3) if len(cov) else None,
            'extrait_ecart_median_m': pct(e, 50), 'extrait_ecart_p90_m': pct(e, 90),
            'grossier_ecart_median_m': pct(co, 50), 'grossier_ecart_p90_m': pct(co, 90),
            'grossier_sup_10m_pct': round(100 * float((co > 10).mean()), 1) if len(co) else None,
            'final_ecart_median_m': pct(fi, 50), 'final_ecart_p90_m': pct(fi, 90),
            'final_sup_10m_pct': round(100 * float((fi > 10).mean()), 1) if len(fi) else None,
        }
    json.dump(report, open(out / 'evaluation.json', 'w'), indent=1, ensure_ascii=False)
    log(json.dumps(report, indent=1, ensure_ascii=False))


def vertex_check(coords_lonlat, coarse_merc, source, props, vmax_m=15.0, gmed_m=15.0):
    """False-positive filter (calibrated on lines closed before 1950, Brittany).

    The vertices of hand-drawn coarse traces are precise (1-3 m): a correct piece passes within
    vmax_m of every coarse vertex it covers; with no vertex covered, it must stay
    within gmed_m (median) of the coarse trace. SNCF shapes (generalised vertices):
    confidence >= 0.8 required instead.
    Returns (keep, number of vertices covered, max offset from the vertices in m or None).
    """
    ll = np.asarray(coords_lonlat)
    mx, my = lonlat_to_merc(ll[:, 0], ll[:, 1])
    s = merc_scale(float(ll[:, 1].mean()))
    e = LineString(np.stack([mx, my], 1))
    co = LineString(coarse_merc)
    V = shapely.points(np.asarray(coarse_merc))
    a, b = sorted(shapely.line_locate_point(co, shapely.points(np.asarray(e.coords)[[0, -1]])))
    sv = shapely.line_locate_point(co, V)
    inside = (sv > a + 20 * s) & (sv < b - 20 * s)
    nv = int(inside.sum())
    vmax = float((shapely.distance(V[inside], e) / s).max()) if nv else None
    if source == 'sncf':
        # Generalised SNCF shapes: real offset up to ~165 m (measured); beyond that, it is no longer the same track.
        return props['confidence'] >= 0.8 and props['ecart_grossier_med_m'] <= 165, nv, vmax
    if nv:
        return vmax <= vmax_m, nv, vmax
    return props['ecart_grossier_med_m'] <= gmed_m, nv, vmax


def sag_check(coords_lonlat, coarse_merc, base_m=15.0, r_min=200.0, cap_m=80.0, step_m=10.0):
    """Shape check: allowed offset from the coarse trace = base + sagitta of a curve of radius
    >= r_min over the local chord (c² / 8R, capped), distributed as a parabola between the vertices.
    Returns the share of the piece's points outside this envelope."""
    ll = np.asarray(coords_lonlat)
    mx, my = lonlat_to_merc(ll[:, 0], ll[:, 1])
    s = merc_scale(float(ll[:, 1].mean()))
    e = LineString(np.stack([mx, my], 1))
    C = np.asarray(coarse_merc)
    co = LineString(C)
    pts = shapely.line_interpolate_point(e, np.linspace(0, e.length, max(2, int(e.length / (step_m * s)) + 1)))
    d = shapely.distance(pts, co) / s
    along = shapely.line_locate_point(co, pts)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(C, axis=0).T))])
    i = np.clip(np.searchsorted(cum, along, side='right') - 1, 0, len(C) - 2)
    c = (cum[i + 1] - cum[i]) / s
    t = np.where(c > 0, (along - cum[i]) / np.maximum(cum[i + 1] - cum[i], 1e-9), 0.5)
    allowed = base_m + np.minimum(c ** 2 / (8 * r_min), cap_m) * 4 * t * (1 - t)
    return float((d > allowed).mean())


def write(args, log=print):
    """Writes the extracted pieces ('coarse' runs, comma-separated) to source_traces.

    Replaces, for the layer, all the 'ign-historique' pieces of the lines present in the runs.
    The runs must cover disjoint areas (portions selected by centroid).
    """
    feats, summary, ev, runs, processed = [], None, {}, args.run.split(','), set()
    for run in runs:
        out = Path(args.work) / 'runs' / run
        sm = json.load(open(out / 'summary.json'))
        if sm['targets'] == 'jonction':
            from .jwrite import write_junctions
            return write_junctions(args, log=log)
        if sm['targets'] != 'coarse':
            raise SystemExit(f'{run}: only "coarse" or "jonction" runs are written to the database')
        if summary and sm['layer'] != summary['layer']:
            raise SystemExit('runs of different layers')
        summary = summary or sm
        coarse = {}
        for line in open(out / 'targets.jsonl'):
            t = json.loads(line)
            coarse[t['tid']] = (t['coarse'], t['source'])
            processed.add(int(t['line_id']))
        n_all = n_conf = 0
        for f in json.load(open(out / 'extracted.geojson'))['features']:
            p = f['properties']
            n_all += 1
            if p['confidence'] < args.min_conf:
                continue
            n_conf += 1
            keep, nv, vmax = vertex_check(f['geometry']['coordinates'], *coarse[p['tid']], p)
            out_env = sag_check(f['geometry']['coordinates'], coarse[p['tid']][0]) if coarse[p['tid']][1] != 'sncf' else 0.0
            keep = keep and out_env <= 0.05
            p.update({'run': run, 'sommets_couverts': nv, 'sommets_ecart_max_m': None if vmax is None else round(vmax, 1),
                      'hors_enveloppe_corde_pct': round(100 * out_env, 1)})
            if keep or args.no_vertex_check:
                feats.append(f)
        log(f'  {run}: {n_all} pieces, {n_conf} with confidence >= {args.min_conf}, {len(feats)} kept in total')
        if (out / 'evaluation-ref.json').exists():
            ev = json.load(open(out / 'evaluation-ref.json'))
    if not ev and getattr(args, 'eval_ref', None):
        ev = json.load(open(args.eval_ref))
    layer = summary['layer']
    lines = sorted({f['properties']['line_id'] for f in feats})
    # Replacement: all the lines processed by the runs (including those with no piece kept).
    processed_lines = sorted(processed | set(lines))
    log(f'{len(feats)} pieces (confidence >= {args.min_conf}) on {len(lines)} lines, '
        f'{sum(f["properties"]["len_m"] for f in feats) / 1000:.1f} km')
    if args.dry_run:
        return
    conn = vdb.connect()
    with conn.transaction():
        if getattr(args, 'replace_all', False):
            n = conn.execute("DELETE FROM source_traces WHERE source = 'ign-historique' AND layer = %s AND match_method <> 'ml-jonction'", (layer,)).rowcount
        else:
            n = conn.execute("DELETE FROM source_traces WHERE source = 'ign-historique' AND layer = %s AND line_id = ANY(%s) AND match_method <> 'ml-jonction'",
                             (layer, processed_lines)).rowcount
        log(f'  {n} old pieces replaced')
        counters = {}
        for f in feats:
            p = f['properties']
            k = counters[p['tid']] = counters.get(p['tid'], 0) + 1
            ref = f"{layer}:{p['line_id']}:{p['part_id']}:{p['tid'].split('-')[-1]}:{k}"[:64]
            props = {**p, 'wmts_layer': summary['wmts'], 'attribution': summary['attribution'], 'licence': 'Licence Ouverte 2.0',
                     'model': summary['model'], 'zoom': summary['zoom'],
                     'ecart_estime_median_m': ev.get('extrait_ecart_median_m'), 'ecart_estime_p90_m': ev.get('extrait_ecart_p90_m'),
                     'evaluation': ev.get('source'), 'methode': 'segmentation U-Net + chemin de coût minimal dans le corridor'}
            wkt = 'MULTILINESTRING((' + ', '.join(f'{x} {y}' for x, y in f['geometry']['coordinates']) + '))'
            conn.execute("""
                INSERT INTO source_traces (source, source_ref, layer, name, properties, length_m, line_id, match_method, match_score,
                                           seen_at, created_at, updated_at, geom)
                VALUES ('ign-historique', %s, %s, %s, %s, %s, %s, 'ml-corridor', %s, NOW(), NOW(), NOW(),
                        ST_GeomFromText(%s, 4326))
                ON CONFLICT (source, source_ref) DO UPDATE SET properties = EXCLUDED.properties, length_m = EXCLUDED.length_m,
                    line_id = EXCLUDED.line_id, match_score = EXCLUDED.match_score, seen_at = NOW(), updated_at = NOW(), geom = EXCLUDED.geom
            """, (ref, layer, summary['attribution'], json.dumps(props, ensure_ascii=False), int(round(p['len_m'])), p['line_id'],
                  p['confidence'], wkt))
    log(f'done: {len(feats)} pieces written to source_traces (source = ign-historique, layer = {layer})')
