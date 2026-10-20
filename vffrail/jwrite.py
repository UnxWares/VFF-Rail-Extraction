"""JUNCTION pieces: filtering, cut at 15 m from the partner track, reconciliation with the
national pieces, evaluation and writing to source_traces (match_method = 'ml-jonction').

Agreed interface (2026-10-20): geometry = axis extracted from the imprecise branch,
stopped at the first approach within 15 m of the precise partner track, NOT stuck onto the track;
properties.jonction = {jid, kind, relation, partner_line_id, partner_source, raccord [lon, lat] | None,
raccord_ecart_m, cap_voie_deg, cap_branche_deg, tolerance_m, precision_m {median, p90}, raccords [...]}.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import shapely
from shapely import LineString

from .geo import lonlat_to_merc, merc_scale, merc_to_lonlat
from .snap import TOL_M, cut_at_partners, merge_with_existing, partners_for

TIP_EPS_M = 1.0


def _ll(m):
    m = np.asarray(m, float).reshape(-1, 2)
    lon, lat = merc_to_lonlat(m[:, 0], m[:, 1])
    return np.round(np.stack([lon, lat], 1), 7)


def _load(run_dir: Path):
    targets = {}
    for row in open(run_dir / 'targets.jsonl'):
        t = json.loads(row)
        targets[t['tid']] = t
    feats = json.load(open(run_dir / 'extracted.geojson'))['features']
    return targets, feats


def junction_pieces(feat, target, min_conf=0.7, eval_mode=False, tol_m=TOL_M):
    """One extracted piece -> junction pieces (Mercator coords, connections) after the filters of the national
    extraction (confidence, vertices, chord envelope) and the cut at tol_m from the partner tracks."""
    from .extract import sag_check, vertex_check

    p = feat['properties']
    coords = feat['geometry']['coordinates']
    if p['confidence'] < min_conf:
        return [], 'confiance'
    keep, nv, vmax = vertex_check(coords, target['coarse'], target['source'], p)
    if not keep:
        return [], 'sommets'
    if target['source'] != 'sncf' and sag_check(coords, target['coarse']) > 0.05:
        return [], 'enveloppe'
    ll = np.asarray(coords)
    mx, my = lonlat_to_merc(ll[:, 0], ll[:, 1])
    s = merc_scale(float(ll[:, 1].mean()))
    partners = partners_for(target)
    if eval_mode:  # the evaluated axis is itself a precise track of the line: only the OTHER lines count
        partners = [q for q in partners if q['line_id'] != target['line_id']]
    jpts = [j['pt'] for j in target.get('junctions', [])]
    pieces = cut_at_partners(np.stack([mx, my], 1), s, partners, jpts, tol_m=tol_m)
    return [(c, r, s) for c, r in pieces], None if pieces else 'coupe'


def _jonction_props(rac, target, precision, s_tip=None):
    js = {j['jid']: j for j in target.get('junctions', [])}
    out = []
    for r in rac:
        out.append({'bout': r['bout'], 'raccord': _ll(r['raccord_merc'])[0].tolist(), 'raccord_ecart_m': r['raccord_ecart_m'],
                    'cap_voie_deg': r['cap_voie_deg'], 'cap_branche_deg': r['cap_branche_deg'],
                    'partner_line_id': r['partner_line_id'], 'partner_source': r['partner_source'],
                    'jonction_dist_m': r['jonction_dist_m']})
    prim = min(out, key=lambda r: r['jonction_dist_m'] if r['jonction_dist_m'] is not None else 1e9) if out else None
    # junction place closest to the main connection (or to the middle of the corridor)
    jid = kind = relation = None
    if js:
        ref = prim['raccord'] if prim else None
        best = None
        for j in js.values():
            d = 0.0 if ref is None else float(np.hypot(*(np.asarray(_ll(j['pt'])[0]) - np.asarray(ref))))
            if best is None or d < best[0]:
                best = (d, j)
        jid, kind, relation = best[1]['jid'], best[1]['kind'], best[1]['relation']
    return {'jid': jid, 'kind': kind, 'relation': relation,
            'partner_line_id': prim['partner_line_id'] if prim else None,
            'partner_source': prim['partner_source'] if prim else None,
            'raccord': prim['raccord'] if prim else None, 'raccord_ecart_m': prim['raccord_ecart_m'] if prim else None,
            'cap_voie_deg': prim['cap_voie_deg'] if prim else None, 'cap_branche_deg': prim['cap_branche_deg'] if prim else None,
            'tolerance_m': TOL_M, 'precision_m': precision, 'raccords': out}


def write_junctions(args, log=print):
    from . import db as vdb

    runs = args.run.split(',')
    work = Path(args.work)
    ev = json.load(open(args.eval_ref)) if getattr(args, 'eval_ref', None) else {}
    precision = {'median': ev.get('jonction_ecart_median_m', ev.get('filtre_ecart_median_m', ev.get('extrait_ecart_median_m'))),
                 'p90': ev.get('jonction_ecart_p90_m', ev.get('filtre_ecart_p90_m', ev.get('extrait_ecart_p90_m'))),
                 'source': ev.get('source')}
    new, reasons, summary, lines_done = [], {}, None, set()
    for run in runs:
        rd = work / 'runs' / run
        sm = json.load(open(rd / 'summary.json'))
        if sm['targets'] != 'jonction':
            raise SystemExit(f'{run}: run "{sm["targets"]}", expected "jonction"')
        if summary and sm['layer'] != summary['layer']:
            raise SystemExit('runs of different layers')
        summary = summary or sm
        targets, feats = _load(rd)
        lines_done |= {t['line_id'] for t in targets.values()}
        for f in feats:
            t = targets[f['properties']['tid']]
            pieces, why = junction_pieces(f, t, args.min_conf)
            reasons[why or 'garde'] = reasons.get(why or 'garde', 0) + 1
            for c, rac, s in pieces:
                new.append({'merc': c, 'rac': rac, 's': s, 'target': t, 'props': f['properties'], 'run': run,
                            '_tips': {'debut': np.asarray(c[0]), 'fin': np.asarray(c[-1])}})
    layer = summary['layer'] if summary else None
    log(f'{len(new)} junction pieces after cutting; reasons: {reasons}')
    conn = vdb.connect()
    # Existing national pieces (same layer, same lines).
    rows = conn.execute("""SELECT id, line_id, ST_AsBinary(ST_Transform(geom, 3857)) FROM source_traces
                           WHERE source = 'ign-historique' AND layer = %s AND match_method = 'ml-corridor' AND line_id = ANY(%s)""",
                        (layer, sorted({n['target']['line_id'] for n in new}))).fetchall()
    existing = {}
    for r, g in zip(rows, vdb.geoms(rows, 2)):
        for ls in vdb.lines_of(g):
            existing.setdefault(int(r[1]), []).append((int(r[0]), ls))
    drop_ids, final = set(), []
    by_line = {}
    for n in new:
        by_line.setdefault(n['target']['line_id'], []).append(n)
    for lid, items in by_line.items():
        ex = existing.get(lid, [])
        s = items[0]['s']
        drop, rest = merge_with_existing([n['merc'] for n in items], [e[1] for e in ex], s)
        drop_ids |= {ex[i][0] for i in drop}
        for n, parts in zip(items, rest):
            for c in parts:
                L = LineString(c).length / n['s']
                # a connection only survives if the cut end is still the end of the piece
                rac = [r for r in n['rac'] if _keeps_tip(c, n['_tips'][r['bout']], n['s'])]
                if L < (50.0 if rac else 200.0):
                    continue
                final.append({**n, 'merc': c, 'rac': rac, 'len_m': L})
    covered = {}
    for n in final:
        jp = _jonction_props(n['rac'], n['target'], precision)
        n['jonction'] = jp
        if jp['raccord'] is not None and jp['jid']:
            covered.setdefault(jp['jid'], []).append(n)
    km = sum(n['len_m'] for n in final) / 1000
    log(f'{len(final)} pieces ({km:.1f} km); {len(covered)} junctions with a connection; '
        f'{len(drop_ids)} national pieces replaced')
    out = work / 'junctions'
    out.mkdir(exist_ok=True)
    gj = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': _ll(n['merc']).tolist()},
         'properties': {'line_id': n['target']['line_id'], 'len_m': round(n['len_m'], 1), 'confidence': n['props']['confidence'],
                        'jonction': n['jonction']}} for n in final]}
    json.dump(gj, open(out / f'write-preview-{layer}.geojson', 'w'))
    with open(out / f'jonctions-couvertes-{layer}.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['jid', 'kind', 'relation', 'line_id', 'partner_line_id', 'raccord_lon', 'raccord_lat', 'raccord_ecart_m',
                    'cap_voie_deg', 'cap_branche_deg', 'morceaux'])
        for jid, ns in sorted(covered.items()):
            j = ns[0]['jonction']
            w.writerow([jid, j['kind'], j['relation'], ns[0]['target']['line_id'], j['partner_line_id'], j['raccord'][0], j['raccord'][1],
                        j['raccord_ecart_m'], j['cap_voie_deg'], j['cap_branche_deg'], len(ns)])
    if args.dry_run:
        return
    with conn.transaction():
        n_del = conn.execute("""DELETE FROM source_traces WHERE source = 'ign-historique' AND layer = %s
                                AND match_method = 'ml-jonction' AND line_id = ANY(%s)""", (layer, sorted(lines_done))).rowcount
        n_rep = conn.execute("DELETE FROM source_traces WHERE id = ANY(%s) AND match_method = 'ml-corridor'", (sorted(drop_ids),)).rowcount
        log(f'  {n_del} old junction pieces, {n_rep} national pieces replaced')
        counters = {}
        for n in final:
            t, p = n['target'], n['props']
            k = counters[t['part_id']] = counters.get(t['part_id'], 0) + 1
            ref = f"j:{layer}:{t['line_id']}:{t['part_id']}:{k}"[:64]
            props = {'len_m': round(n['len_m'], 1), 'confidence': p['confidence'], 'mean_p': p['mean_p'], 'frac_p50': p['frac_p50'],
                     'ecart_grossier_med_m': p['ecart_grossier_med_m'], 'corridor_m': p['corridor_m'], 'tid': t['tid'],
                     'part_id': t['part_id'], 'coarse_source': t['source'], 'run': n['run'], 'jonction': n['jonction'],
                     'wmts_layer': summary['wmts'], 'attribution': summary['attribution'], 'licence': 'Licence Ouverte 2.0',
                     'model': summary['model'], 'zoom': summary['zoom'],
                     'ecart_estime_median_m': precision['median'], 'ecart_estime_p90_m': precision['p90'],
                     'methode': 'segmentation U-Net + chemin de coût minimal dans le corridor de jonction, coupé à 15 m de la voie précise'}
            ll = _ll(n['merc'])
            wkt = 'MULTILINESTRING((' + ', '.join(f'{x} {y}' for x, y in ll) + '))'
            conn.execute("""
                INSERT INTO source_traces (source, source_ref, layer, name, properties, length_m, line_id, match_method, match_score,
                                           seen_at, created_at, updated_at, geom)
                VALUES ('ign-historique', %s, %s, %s, %s, %s, %s, 'ml-jonction', %s, NOW(), NOW(), NOW(), ST_GeomFromText(%s, 4326))
                ON CONFLICT (source, source_ref) DO UPDATE SET properties = EXCLUDED.properties, length_m = EXCLUDED.length_m,
                    line_id = EXCLUDED.line_id, match_score = EXCLUDED.match_score, match_method = EXCLUDED.match_method,
                    seen_at = NOW(), updated_at = NOW(), geom = EXCLUDED.geom
            """, (ref, layer, summary['attribution'], json.dumps(props, ensure_ascii=False), int(round(n['len_m'])), t['line_id'],
                  p['confidence'], wkt))
    log(f'done: {len(final)} junction pieces written (source_traces, ign-historique, {layer}, ml-jonction)')


def _keeps_tip(c, tip, s) -> bool:
    """Does the remaining piece (after reconciliation) still have the original cut end as its end?"""
    c = np.asarray(c)
    return min(float(np.hypot(*(c[0] - tip))), float(np.hypot(*(c[-1] - tip)))) / s <= TIP_EPS_M


def evaluate_junctions(args, log=print):
    """Accuracy AT JUNCTIONS ('jonction-eval' run): offset of the cut pieces to the true axis, and of the cut end."""
    rd = Path(args.work) / 'runs' / args.run
    targets, feats = _load(rd)
    d_all, d_tip, L_keep, L_raw, reasons = [], [], 0.0, 0.0, {}
    for f in feats:
        t = targets[f['properties']['tid']]
        if not t.get('truth'):
            continue
        ll = np.asarray(f['geometry']['coordinates'])
        s = merc_scale(float(ll[:, 1].mean()))
        L_raw += f['properties']['len_m']
        pieces, why = junction_pieces(f, t, 0.7, eval_mode=True)
        reasons[why or 'garde'] = reasons.get(why or 'garde', 0) + 1
        truth = shapely.MultiLineString([LineString(a) for a in t['truth']])
        for c, rac, s in pieces:
            e = LineString(c)
            n = max(2, int(e.length / (10 * s)) + 1)
            d = shapely.distance(shapely.line_interpolate_point(e, np.linspace(0, e.length, n)), truth) / s
            d_all.extend(d.tolist())
            L_keep += e.length / s
            for r in rac:
                tip = c[0] if r['bout'] == 'debut' else c[-1]
                d_tip.append(float(shapely.distance(shapely.Point(tip), truth)) / s)

    def pct(a, q):
        return round(float(np.percentile(a, q)), 1) if len(a) else None

    rep = {'source': f'{args.run} (jonctions, axe vrai BD TOPO / OSM)', 'km_bruts': round(L_raw / 1000, 1),
           'km_gardes': round(L_keep / 1000, 1), 'motifs': reasons,
           'jonction_ecart_median_m': pct(d_all, 50), 'jonction_ecart_p90_m': pct(d_all, 90),
           'jonction_faux_sup_20m_pct': round(100 * float((np.asarray(d_all) > 20).mean()), 1) if d_all else None,
           'bout_coupe_n': len(d_tip), 'bout_coupe_ecart_median_m': pct(d_tip, 50), 'bout_coupe_ecart_p90_m': pct(d_tip, 90)}
    json.dump(rep, open(rd / 'evaluation-jonctions.json', 'w'), indent=1, ensure_ascii=False)
    log(json.dumps(rep, indent=1, ensure_ascii=False))
    return rep
