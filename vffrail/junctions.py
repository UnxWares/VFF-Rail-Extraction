"""Junctions between IMPRECISE and PRECISE traces, and extraction corridors around them.

Imprecise trace: piece of `line_reference_parts` from a coarse source
(config.coarse_sources) and not registered (`matched_source IS NULL`). Precise trace: BD TOPO / OSM piece, or registered on
BD TOPO / OSM / ign-historique ("connector" links do not count).

Places kept (Lambert-93, all eras merged: the reference pieces are shared by the eras):
- 'extremite' (end): end of an imprecise trace within 150 m of a precise trace
  (relation 'autre-ligne' (other line) if a precise trace of ANOTHER line is within 150 m, otherwise 'meme-ligne' (same line));
- 'convergence': stretch of at least 100 m of an imprecise trace within 150 m of a precise trace
  of another line (a line that runs alongside or joins at an angle);
- 'detecteur' (detector): places from the national detector (CSV lon, lat), attached to the closest imprecise trace (<= 150 m).

Corridor: 1.5 km along the imprecise branch from the end; ± 750 m around a
convergence zone or a detector place; intervals merged per piece. Each corridor becomes an
extraction target (`targets.jsonl`, EPSG:3857 coordinates); the precise tracks of the other lines are only
"released" (path allowed) there within 60 m of the junction points, not at arbitrary corridor ends.

'eval' mode: same rules on the coarse pieces REGISTERED on BD TOPO / OSM (true axis known),
with the original coarse trace as the corridor: measures the accuracy AT JUNCTIONS.
"""

from __future__ import annotations

import csv
import glob
import json
import random
from pathlib import Path

import numpy as np
import shapely
import shapely.ops
from shapely import LineString

from . import config
from . import db as vdb

CORR_END_M = 1500.0
CORR_HALF_M = 750.0
NEAR_M = 150.0
MIN_CONV_M = 100.0

IMP_SQL = {
    'coarse': f"""p.source IN {config.coarse_sql_list()} AND p.matched_source IS NULL""",
    'eval': f"""p.source IN {config.coarse_sql_list()} AND p.matched_source IN ('bdtopo','osm') AND p.length_m >= 300""",
}


def _setup(conn, mode):
    conn.execute('DROP TABLE IF EXISTS jx_imp; DROP TABLE IF EXISTS jx_pre')
    if mode == 'coarse':
        geom = 'd.geom'
        src = 'NULL::geometry'
    else:
        # original coarse trace under the registered piece (as in extract.fetch_targets 'eval')
        geom = """(SELECT ST_LineMerge(ST_CollectionExtract(ST_Intersection(ST_Collect(t.geom),
                       ST_Transform(ST_Buffer(ST_Transform(d.geom, 2154), 150), 4326)), 2))
                   FROM source_traces t WHERE t.id IN (SELECT jsonb_array_elements_text(p.source_trace_ids)::bigint))"""
        src = 'ST_Transform(d.geom, 2154)'
    conn.execute(f"""
        CREATE TEMP TABLE jx_imp AS
        SELECT p.id AS part_id, p.line_id, p.source, d.path[1] AS k, l.network,
               ST_Transform(g.g, 2154) AS g, {src} AS truth
        FROM line_reference_parts p
        JOIN lines l ON l.id = p.line_id
        CROSS JOIN LATERAL ST_Dump(p.geom) d
        CROSS JOIN LATERAL (SELECT {geom} AS g) g
        WHERE {IMP_SQL[mode]}
    """)
    conn.execute("DELETE FROM jx_imp WHERE g IS NULL OR ST_IsEmpty(g) OR GeometryType(g) <> 'LINESTRING' OR ST_Length(g) < 150")
    conn.execute("""
        CREATE TEMP TABLE jx_pre AS
        SELECT p.id AS part_id, p.line_id, COALESCE(p.matched_source, p.source) AS src, ST_Transform(p.geom, 2154) AS g
        FROM line_reference_parts p
        WHERE (p.source IN ('bdtopo','osm','contrib') AND p.matched_source IS NULL)
           OR p.matched_source IN ('bdtopo','osm','contrib','ign-historique')
    """)
    conn.execute('CREATE INDEX ON jx_pre USING gist (g); CREATE INDEX ON jx_imp USING gist (g); ANALYZE jx_pre; ANALYZE jx_imp')


def _places(conn, mode, detector_csvs):
    """Junction places: (part_id, k, s0_m, s1_m, junction_s_m list, props)."""
    places = []
    # Free ends.
    rows = conn.execute(f"""
        WITH e AS (
            SELECT part_id, line_id, k, 0 AS fr, ST_StartPoint(g) AS pt, ST_Length(g) AS L FROM jx_imp
            UNION ALL SELECT part_id, line_id, k, 1, ST_EndPoint(g), ST_Length(g) FROM jx_imp)
        SELECT e.part_id, e.line_id, e.k, e.fr, e.L, q.other, q.d_other, q.d_any, q.partners,
               ST_X(ST_Transform(e.pt, 4326)), ST_Y(ST_Transform(e.pt, 4326)),
               ST_AsBinary(ST_Transform(ST_Transform(a.cp, 4326), 3857))
        FROM e CROSS JOIN LATERAL (
            SELECT bool_or(q.line_id <> e.line_id) AS other,
                   min(ST_Distance(q.g, e.pt)) FILTER (WHERE q.line_id <> e.line_id) AS d_other,
                   min(ST_Distance(q.g, e.pt)) AS d_any,
                   array_agg(DISTINCT q.line_id) FILTER (WHERE q.line_id <> e.line_id) AS partners
            FROM jx_pre q WHERE ST_DWithin(q.g, e.pt, {NEAR_M})) q
        CROSS JOIN LATERAL (
            /* anchor point: closest on the partner precise track (other line first) */
            SELECT ST_ClosestPoint(q2.g, e.pt) AS cp FROM jx_pre q2
            WHERE ST_DWithin(q2.g, e.pt, {NEAR_M}) AND (NOT q.other OR q2.line_id <> e.line_id)
            ORDER BY q2.g <-> e.pt LIMIT 1) a
        WHERE q.d_any IS NOT NULL
    """).fetchall()
    for row in rows:
        part_id, line_id, k, fr, L, other, d_other, d_any, partners, lon, lat = row[:11]
        anchor = vdb.geoms([row], 11)[0]
        s = 0.0 if fr == 0 else L
        places.append({'part_id': part_id, 'k': k, 's0': max(0.0, s - CORR_END_M) if fr else 0.0,
                       's1': L if fr else min(L, CORR_END_M), 'js': [s], 'kind': 'extremite',
                       'relation': 'autre-ligne' if other else 'meme-ligne', 'line_id': line_id,
                       'dist_m': round(float(d_other if other else d_any), 1), 'partners': partners or [],
                       'anchor': [anchor.x, anchor.y] if anchor is not None else None, 'fr': fr,
                       'lon': lon, 'lat': lat})
    # Convergence zones (other line within 150 m over >= 100 m).
    rows = conn.execute(f"""
        WITH n AS (
            SELECT i.part_id, i.k, i.line_id, i.g, ST_Union(ST_Buffer(q.g, {NEAR_M}, 'quad_segs=2')) AS b,
                   array_agg(DISTINCT q.line_id) AS partners
            FROM jx_imp i JOIN jx_pre q ON ST_DWithin(i.g, q.g, {NEAR_M}) AND q.line_id <> i.line_id
            GROUP BY i.part_id, i.k, i.line_id, i.g),
        pc AS (
            SELECT n.part_id, n.k, n.line_id, n.g, n.partners, d.geom AS piece
            FROM n CROSS JOIN LATERAL ST_Dump(ST_CollectionExtract(ST_Intersection(n.g, n.b), 2)) d)
        SELECT part_id, k, line_id, partners, ST_Length(g),
               ST_LineLocatePoint(g, ST_StartPoint(piece)) * ST_Length(g), ST_LineLocatePoint(g, ST_EndPoint(piece)) * ST_Length(g),
               ST_Length(piece), ST_X(ST_Transform(ST_LineInterpolatePoint(piece, 0.5), 4326)),
               ST_Y(ST_Transform(ST_LineInterpolatePoint(piece, 0.5), 4326))
        FROM pc WHERE ST_Length(piece) >= {MIN_CONV_M}
    """).fetchall()
    for part_id, k, line_id, partners, L, a, b, plen, lon, lat in rows:
        a, b = sorted((a, b))
        places.append({'part_id': part_id, 'k': k, 's0': max(0.0, a - CORR_HALF_M), 's1': min(L, b + CORR_HALF_M),
                       'js': [a, b], 'kind': 'convergence', 'relation': 'autre-ligne', 'line_id': line_id,
                       'dist_m': None, 'partners': partners or [], 'conv_m': round(plen), 'lon': lon, 'lat': lat})
    # Places from the national detector.
    pts = []
    for path in detector_csvs:
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                lon = r.get('lon') or r.get('longitude') or r.get('x')
                lat = r.get('lat') or r.get('latitude') or r.get('y')
                try:
                    typ = r.get('typ') or r.get('type') or r.get('defaut') or ''
                    pts.append((float(lon), float(lat), Path(path).name, f"{typ}:{r.get('sub', '')}".strip(':')))
                except (TypeError, ValueError):
                    continue
    if pts:
        conn.execute('DROP TABLE IF EXISTS jx_det; CREATE TEMP TABLE jx_det (lon float8, lat float8, f text, t text)')
        with conn.cursor().copy('COPY jx_det FROM STDIN') as cp:
            for p in pts:
                cp.write_row(p)
        rows = conn.execute(f"""
            SELECT i.part_id, i.k, i.line_id, ST_Length(i.g), ST_LineLocatePoint(i.g, pt) * ST_Length(i.g), d.lon, d.lat, d.f, d.t,
                   ST_Distance(i.g, pt)
            FROM (SELECT *, ST_Transform(ST_SetSRID(ST_MakePoint(lon, lat), 4326), 2154) AS pt FROM jx_det) d
            CROSS JOIN LATERAL (SELECT * FROM jx_imp i WHERE ST_DWithin(i.g, d.pt, {NEAR_M}) ORDER BY i.g <-> d.pt LIMIT 1) i
        """).fetchall()
        for part_id, k, line_id, L, s, lon, lat, fname, typ, d in rows:
            places.append({'part_id': part_id, 'k': k, 's0': max(0.0, s - CORR_HALF_M), 's1': min(L, s + CORR_HALF_M),
                           'js': [s], 'kind': 'detecteur', 'relation': typ or fname, 'line_id': line_id,
                           'dist_m': round(float(d), 1), 'partners': [], 'lon': lon, 'lat': lat})
    return places, len(pts)


def _merge(iv):
    iv = sorted(iv)
    out = []
    for a, b, j in iv:
        if out and a <= out[-1][1] + 50:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2].extend(j)
        else:
            out.append([a, b, list(j)])
    return out


def build(args, log=print):
    out = Path(args.work) / 'junctions'
    out.mkdir(parents=True, exist_ok=True)
    conn = vdb.connect()
    _setup(conn, args.mode)
    dets = [d for d in sorted(glob.glob(args.detector)) if 'test' not in Path(d).name] if args.detector and args.mode == 'coarse' else []
    places, n_det = _places(conn, args.mode, dets)
    log(f'{len(places)} places ({args.mode}); detector: {len(dets)} files, {n_det} points')
    excl = set((args.exclude_networks or '').split(',')) - {''}
    rows = conn.execute("""
        SELECT part_id, k, line_id, source, network, ST_AsBinary(ST_Transform(ST_Transform(g, 4326), 3857)),
               ST_AsBinary(ST_Transform(ST_Transform(truth, 4326), 3857)), ST_Length(g)
        FROM jx_imp
    """).fetchall()
    geo = {(r[0], r[1]): r for r in rows}
    by_part = {}
    for i, p in enumerate(places):
        p['jid'] = f"{args.mode[0]}{i}"
        by_part.setdefault((p['part_id'], p['k']), []).append(p)
    keys = sorted(by_part)
    if args.sample and len(keys) > args.sample:
        # sample (eval mode): junctions with ANOTHER line first
        random.Random(42).shuffle(keys)
        keys = sorted(keys, key=lambda kk: 0 if any(p['relation'] == 'autre-ligne' for p in by_part[kk]) else 1)[:args.sample]
    targets, n_ex = [], 0
    for key in keys:
        part_id, k, line_id, source, network, gw, tw, L2154 = geo[key]
        if network in excl:
            n_ex += 1
            continue
        g = vdb.geoms([geo[key]], 5)[0]
        truth = vdb.geoms([geo[key]], 6)[0]
        line = LineString(np.asarray(g.coords)[:, :2])
        f = line.length / L2154  # Mercator m per Lambert-93 m (~ 1/cos lat)
        pl = {p['jid']: p for p in by_part[key]}
        for a, b, js in _merge([(p['s0'], p['s1'], [(p['jid'], s) for s in p['js']]) for p in by_part[key]]):
            if b - a < 150:
                continue
            sub = shapely.ops.substring(line, a * f, b * f)
            # Free end a few metres from the precise track: the corridor is extended up to it
            # (anchor point), otherwise the extracted track could never get within 15 m of it.
            cs = np.asarray(sub.coords)
            anchors = []
            for jid, sj in js:
                pj = pl[jid]
                if pj['kind'] != 'extremite' or not pj.get('anchor'):
                    continue
                an = np.asarray(pj['anchor'])
                if pj['fr'] == 0 and a <= 1.0 and np.hypot(*(an - cs[0])) > 2 * f:
                    cs = np.vstack([an, cs])
                    anchors.append(an.tolist())
                elif pj['fr'] == 1 and b >= L2154 - 1.0 and np.hypot(*(an - cs[-1])) > 2 * f:
                    cs = np.vstack([cs, an])
                    anchors.append(an.tolist())
            sub = LineString(cs)
            jx = [{'jid': j, 'kind': pl[j]['kind'], 'relation': pl[j]['relation'], 'partners': pl[j]['partners'],
                   'pt': list(line.interpolate(s * f).coords[0])} for j, s in js]
            targets.append({'tid': f'j{part_id}-{k}-{int(a)}', 'part_id': part_id, 'line_id': line_id, 'source': source,
                            'coarse': np.asarray(sub.coords).tolist(), 'release': [j['pt'] for j in jx] + anchors,
                            'junctions': jx, 'precise': [],
                            'truth': [a_.tolist() for a_ in vdb.lines_of(truth)] if truth is not None else None,
                            'len_m': round(b - a)})
    # Precise tracks (all lines, including the targeted line) within 250 m of each corridor: junction registration.
    conn.execute('DROP TABLE IF EXISTS jx_cor; CREATE TEMP TABLE jx_cor (i int, g geometry)')
    with conn.cursor().copy('COPY jx_cor (i, g) FROM STDIN') as cp:
        for i, t in enumerate(targets):
            cp.write_row((i, 'SRID=3857;' + LineString(t['coarse']).wkt))
    conn.execute('UPDATE jx_cor SET g = ST_Transform(g, 2154); CREATE INDEX ON jx_cor USING gist (g); ANALYZE jx_cor')
    rows = conn.execute("""
        SELECT c.i, q.line_id, q.src, q.part_id,
               ST_AsBinary(ST_Transform(ST_CollectionExtract(ST_Intersection(q.g, ST_Buffer(c.g, 250, 'quad_segs=4')), 2), 3857))
        FROM jx_cor c JOIN jx_pre q ON ST_DWithin(q.g, c.g, 250)
    """).fetchall()
    for r, gq in zip(rows, vdb.geoms(rows, 4)):
        for ls in vdb.lines_of(gq):
            if len(ls) >= 2:
                targets[r[0]]['precise'].append({'line_id': r[1], 'src': r[2], 'part_id': r[3], 'coords': np.round(ls, 2).tolist()})
    with open(out / f'targets-{args.name}.jsonl', 'w') as fh:
        for t in targets:
            fh.write(json.dumps(t) + '\n')
    feats = [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [p['lon'], p['lat']]},
              'properties': {k: v for k, v in p.items() if k not in ('lon', 'lat', 's0', 's1', 'js', 'anchor', 'fr')}} for p in places]
    json.dump({'type': 'FeatureCollection', 'features': feats}, open(out / f'junctions-{args.name}.geojson', 'w'))
    stats = {}
    for p in places:
        kk = f"{p['kind']}/{p['relation']}"
        stats[kk] = stats.get(kk, 0) + 1
    km = sum(t['len_m'] for t in targets) / 1000
    log(json.dumps({'lieux': stats, 'cibles': len(targets), 'km_corridors': round(km, 1), 'morceaux_exclus_reseau': n_ex,
                    'lignes': len({t['line_id'] for t in targets})}, ensure_ascii=False))
