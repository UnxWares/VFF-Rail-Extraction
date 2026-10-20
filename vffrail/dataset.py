"""AUTO-LABELLED training set.

Ground truth: PRECISE axis (BD TOPO / OSM / validated correction, including the
coarse pieces already registered on one of these sources) of the lines IN SERVICE in
all the requested eras (e.g. 1940 and 1950 for the "1950" SCAN 50, whose sheets date
from the 1920s to 1950), clipped to 300 m from the in-service segments of these eras
(a part built later is not on the map).

"Ignored" mask (no loss): everything that could be a track without being in the
ground truth (other precise traces within 25 m, coarse traces (imprecise sources)
within 150 m, service tracks, lines closed before the period...). The rest is
negative. The final label (axis ± width, ignored band, shift) is rasterised on the
fly by the training from the axes stored as JSON.

A chip = block of 2 × 2 tiles (512 px) at the working level of the layer.
Validation: spatial blocks of 16 × 16 tiles (~13 km at z15) drawn at random (1 in 5).
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import shapely
from PIL import Image
from shapely import STRtree, box

from . import db as vdb
from .geo import TILE, draw_lines, ground_res, merc_to_lonlat, merc_to_px, px_to_merc
from .tiles import LAYERS, PoliteFetcher, TileCache, blank_ratio

PRECISE = ('bdtopo', 'osm', 'contrib')
CHIP_TILES = 2
CHIP = CHIP_TILES * TILE


def era_ids(conn, years):
    rows = conn.execute('SELECT id, year FROM eras WHERE year = ANY(%s)', (list(years),)).fetchall()
    found = {y: i for i, y in rows}
    missing = [y for y in years if y not in found]
    if missing:
        raise SystemExit(f'unknown eras: {missing}')
    return [found[y] for y in years]


def fetch_truth(conn, bbox, years, margin_deg=0.0):
    """Precise axes of the lines in service in all the requested eras (EPSG:3857).

    Also fills the temporary table rx_truth (line_id, g Lambert-93) for fetch_others().
    """
    ids = era_ids(conn, years)
    lon0, lat0, lon1, lat1 = bbox
    env = vdb.env_sql((lon0 - margin_deg, lat0 - margin_deg, lon1 + margin_deg, lat1 + margin_deg))
    conn.execute('DROP TABLE IF EXISTS rx_truth')
    conn.execute(f"""
        CREATE TEMP TABLE rx_truth AS
        WITH seg AS (
            SELECT line_id, era_id, ST_Union(ST_Buffer(ST_Transform(geom, 2154), 300, 'quad_segs=4')) AS b
            FROM line_segments
            WHERE era_id = ANY(%(eras)s) AND status IN ('in_service', 'single_track') AND geom && {env}
            GROUP BY line_id, era_id
        ),
        zone AS (
            SELECT line_id, array_agg(b) AS bs, COUNT(*) AS n
            FROM seg GROUP BY line_id HAVING COUNT(*) = cardinality(%(eras)s::bigint[])
        ),
        z AS (
            SELECT line_id, CASE WHEN n = 1 THEN bs[1] ELSE ST_Intersection(bs[1], bs[2]) END AS b FROM zone
        )
        SELECT p.line_id, ST_CollectionExtract(ST_Intersection(ST_Transform(p.geom, 2154), z.b), 2) AS g
        FROM line_reference_parts p JOIN z USING (line_id)
        WHERE p.geom && {env}
          AND (p.matched_source IN ('bdtopo', 'osm', 'contrib')
               OR (p.matched_source IS NULL AND p.source IN ('bdtopo', 'osm', 'contrib')))
    """, {'eras': ids})
    rows = conn.execute('SELECT line_id, ST_AsBinary(ST_Transform(g, 3857)) FROM rx_truth WHERE NOT ST_IsEmpty(g)').fetchall()
    out = []
    for (line_id, _), g in zip(rows, vdb.geoms(rows, 1)):
        for ls in vdb.lines_of(g):
            if len(ls) >= 2:
                out.append((line_id, ls))
    return out


def fetch_others(conn, bbox, margin_deg=0.02):
    """All the other known traces (for the "ignored" mask), EPSG:3857.

    The COARSE traces of a ground-truth line are only kept beyond 80 m from its
    precise axis: their real position is known, and the roads running alongside the
    track must remain negative examples.
    """
    lon0, lat0, lon1, lat1 = bbox
    env = vdb.env_sql((lon0 - margin_deg, lat0 - margin_deg, lon1 + margin_deg, lat1 + margin_deg))
    rows = conn.execute(f"""
        WITH t AS (SELECT line_id, ST_Buffer(ST_Collect(g), 80, 'quad_segs=4') AS b FROM rx_truth GROUP BY line_id)
        SELECT st.source, ST_AsBinary(ST_Transform(
                   CASE WHEN t.b IS NOT NULL AND st.source NOT IN ('bdtopo', 'osm', 'contrib')
                        THEN ST_Difference(ST_Transform(st.geom, 2154), t.b)
                        ELSE ST_Transform(st.geom, 2154) END, 3857))
        FROM source_traces st LEFT JOIN t ON t.line_id = st.line_id
        WHERE st.geom && {env} AND st.source <> 'ign-historique'
    """).fetchall()
    out = []
    for (src, _), g in zip(rows, vdb.geoms(rows, 1)):
        for ls in vdb.lines_of(g):
            out.append((src in PRECISE, ls))
    return out


def chip_fold(z, bx, by, val_every=5) -> str:
    sx, sy = (bx * CHIP_TILES) >> 4, (by * CHIP_TILES) >> 4
    h = int(hashlib.md5(f'{z}:{sx}:{sy}'.encode()).hexdigest()[:8], 16)
    return 'val' if h % val_every == 0 else 'train'


def chip_blocks_for_lines(lines_merc, z, min_len_px=30):
    """Blocks (bx, by) crossed by the lines, with the length (px) inside."""
    acc = {}
    for ls in lines_merc:
        px, py = merc_to_px(ls[:, 0], ls[:, 1], z)
        for i in range(len(px) - 1):
            seg_len = math.hypot(px[i + 1] - px[i], py[i + 1] - py[i])
            n = max(1, int(seg_len / 16))
            for k in range(n):
                t = (k + 0.5) / n
                bx = int((px[i] + t * (px[i + 1] - px[i])) // CHIP)
                by = int((py[i] + t * (py[i + 1] - py[i])) // CHIP)
                acc[(bx, by)] = acc.get((bx, by), 0.0) + seg_len / n
    return {k: v for k, v in acc.items() if v >= min_len_px}


# ===== parallel rasterisation (fork: the trees are shared) ========================
_G = {}


def _init(truth, others, z, root, layer_key, out):
    _G['truth'] = truth
    _G['ttree'] = STRtree([shapely.LineString(ls) for _, ls in truth])
    _G['others'] = others
    _G['otree'] = STRtree([shapely.LineString(ls) for _, ls in others])
    _G['z'] = z
    _G['cache'] = TileCache(root, LAYERS[layer_key])
    _G['out'] = Path(out)


def _make_chip(args):
    bx, by, fold = args
    z = _G['z']
    cache = _G['cache']
    img, valid = cache.mosaic(z, bx * CHIP_TILES, by * CHIP_TILES, CHIP_TILES, CHIP_TILES)
    if valid.mean() < 0.99 or blank_ratio(img) > 0.6:
        return None
    x0, y0 = bx * CHIP, by * CHIP
    mx0, my0 = px_to_merc(x0, y0, z)
    mx1, my1 = px_to_merc(x0 + CHIP, y0 + CHIP, z)
    lon, lat = merc_to_lonlat((mx0 + mx1) / 2, (my0 + my1) / 2)
    gres = ground_res(z, float(lat))
    margin = 400.0  # Mercator m (~270 m on the ground): "ignored" buffers overflowing the edge
    q = box(mx0 - margin, my1 - margin, mx1 + margin, my0 + margin)

    def to_px(ls):
        px, py = merc_to_px(ls[:, 0], ls[:, 1], z)
        return np.stack([px - x0, py - y0], axis=1)

    truth = []
    for i in _G['ttree'].query(q):
        line_id, ls = _G['truth'][i]
        truth.append({'line_id': int(line_id), 'pts': np.round(to_px(ls), 1).tolist()})
    ign = Image.new('L', (CHIP, CHIP), 0)
    precise, coarse = [], []
    for i in _G['otree'].query(q):
        is_precise, ls = _G['others'][i]
        (precise if is_precise else coarse).append(to_px(ls))
    draw_lines((CHIP, CHIP), coarse, 2 * 150 / gres, 255, ign)
    draw_lines((CHIP, CHIP), precise, 2 * 25 / gres, 255, ign)
    name = f'{z}_{bx}_{by}'
    Image.fromarray(img).save(_G['out'] / 'img' / f'{name}.png', optimize=False, compress_level=1)
    ign.save(_G['out'] / 'ign' / f'{name}.png')
    return {'id': name, 'z': z, 'bx': bx, 'by': by, 'fold': fold, 'lat': round(float(lat), 5), 'lon': round(float(lon), 5),
            'gres': round(gres, 3), 'truth': truth}


def build(args, log=print):
    layer = LAYERS[args.layer]
    z = args.zoom or layer.zoom
    out = Path(args.work) / 'datasets' / f'{layer.key}-{args.name}'
    (out / 'img').mkdir(parents=True, exist_ok=True)
    (out / 'ign').mkdir(parents=True, exist_ok=True)
    conn = vdb.connect()
    years = [int(y) for y in args.eras.split(',')]
    log(f'ground truth: lines in service in eras {years} within {args.bbox}')
    truth = fetch_truth(conn, args.bbox, years)
    others = fetch_others(conn, args.bbox)
    log(f'  {len(truth)} precise axes, {len(others)} other traces (ignored mask)')

    blocks = chip_blocks_for_lines([ls for _, ls in truth], z)
    keys = sorted(blocks)
    random.Random(42).shuffle(keys)
    if args.max_chips and len(keys) > args.max_chips:
        keys = keys[:args.max_chips]
    # A few random "negative" chips in the area (towns, roads, rivers).
    lon0, lat0, lon1, lat1 = args.bbox
    from .geo import lonlat_to_merc
    (ax, ay), (bx_, by_) = [merc_to_px(*lonlat_to_merc(a, b), z) for a, b in ((lon0, lat1), (lon1, lat0))]
    rng = random.Random(7)
    neg = set()
    for _ in range(int(len(keys) * args.neg_ratio)):
        k = (int(rng.uniform(ax, bx_) // CHIP), int(rng.uniform(ay, by_) // CHIP))
        if k not in blocks:
            neg.add(k)
    keys += sorted(neg)
    log(f'  {len(keys)} chips ({len(neg)} random negatives)')

    tiles = sorted({(bx * CHIP_TILES + i, by * CHIP_TILES + j) for bx, by in keys for i in range(CHIP_TILES) for j in range(CHIP_TILES)})
    cache = TileCache(args.tiles, layer)
    PoliteFetcher(cache, workers=args.dl_workers, max_rps=args.max_rps, log=log).fetch_all(tiles, z)

    jobs = [(bx, by, chip_fold(z, bx, by)) for bx, by in keys]
    ctx = mp.get_context('fork')
    meta = []
    with ctx.Pool(args.workers, initializer=_init, initargs=(truth, others, z, args.tiles, layer.key, out)) as pool:
        for m in pool.imap_unordered(_make_chip, jobs, chunksize=8):
            if m:
                meta.append(m)
    meta.sort(key=lambda m: m['id'])
    with open(out / 'meta.jsonl', 'w') as f:
        for m in meta:
            f.write(json.dumps(m) + '\n')
    n_val = sum(m['fold'] == 'val' for m in meta)
    log(f'done: {len(meta)} chips ({n_val} in validation) -> {out}')
