"""Runs vffrail.precise on a junction run (probabilities on disk, no model, database read-only).

    py.sh -m vffrail.precise_run eval  --run jonctions-eval-gpu --out DIR [--limit N] [--offset K]
    py.sh -m vffrail.precise_run apply --run jonctions-fr-ortho-1950 --out DIR [--limit N]

Output: DIR/<run>-stations.npz (one row per station: target, branch, estimated error, true offset in eval, zone),
DIR/<run>-branches.jsonl (lon/lat geometries, errors, switches).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import numpy as np

from . import config
from . import precise as P
from .geo import lonlat_to_merc, merc_scale, merc_to_lonlat
from .snap import partners_for

WORK = config.work_dir()


def load_run(run):
    rd = WORK / 'runs' / run
    targets = {}
    for row in open(rd / 'targets.jsonl'):
        t = json.loads(row)
        targets[t['tid']] = t
    feats = {}
    for f in json.load(open(rd / 'extracted.geojson'))['features']:
        feats.setdefault(f['properties']['tid'], []).append(f)
    return targets, feats


def networks(line_ids):
    from . import db as vdb

    conn = vdb.connect()
    rows = conn.execute('SELECT id, network, gauge_mm_default FROM lines WHERE id = ANY(%s)', (sorted(line_ids),)).fetchall()
    return {int(r[0]): (r[1], r[2]) for r in rows}


def run_target(t, fl, nets, eval_mode, prob_root):
    c = np.asarray(t['coarse'])
    lat = float(merc_to_lonlat(c[:, 0].mean(), c[:, 1].mean())[1])
    s = merc_scale(lat)
    fg = []
    for f in fl:
        ll = np.asarray(f['geometry']['coordinates'])
        mx, my = lonlat_to_merc(ll[:, 0], ll[:, 1])
        fg.append(np.stack([mx, my], 1) / s)
    allm = np.vstack([c] + [g * s for g in fg])
    m = 400.0 * s
    mos = P.Mosaic(prob_root, (allm[:, 0].min() - m, allm[:, 1].min() - m, allm[:, 0].max() + m, allm[:, 1].max() + m), lat)
    own = t['line_id']
    partners = partners_for(t)
    if eval_mode:
        partners = [q for q in partners if q['line_id'] != own]
    pg = [{'g': np.asarray(q['coords']) / s, 'line_id': q['line_id'], 'part_id': q['part_id'], 'src': q['src']} for q in partners]
    reg = [np.asarray(q['coords']) / s for q in t['precise'] if not (eval_mode and q['line_id'] == own)]
    truth = [np.asarray(a) / s for a in t['truth']] if (eval_mode and t.get('truth')) else None
    net, gauge = nets.get(own, (None, None))
    rmin = P.min_radius(net, gauge)
    oth = [np.asarray(q['coords']) / s for q in t['precise'] if q['line_id'] != own]
    brs, obs = P.process_target(t, fg, mos, pg, reg, rmin, truth, other_g=oth, coarse_g=c / s)
    return brs, obs, s, rmin, net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['eval', 'apply'])
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--tids', default='')
    ap.add_argument('--prob', default=str(WORK / 'prob' / 'ortho-v1'))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    targets, feats = load_run(a.run)
    tids = [x for x in sorted(targets) if x in feats]
    if a.tids:
        want = set(a.tids.split(','))
        tids = [x for x in tids if x in want]
    tids = tids[a.offset:]
    if a.limit:
        tids = tids[:a.limit]
    nets = networks({targets[x]['line_id'] for x in tids})
    eval_mode = a.mode == 'eval'
    st_rows, n_obs = [], []
    tag = f'{a.run}-{a.offset}' if (a.offset or a.limit) else a.run
    bf = open(out / f'{tag}-branches.jsonl', 'w')
    for i, tid in enumerate(tids):
        t = targets[tid]
        try:
            brs, obs, s, rmin, net = run_target(t, feats[tid], nets, eval_mode, a.prob)
        except Exception as ex:  # noqa: BLE001
            print(f'{tid}: ERROR {ex!r}', file=sys.stderr)
            continue
        n_obs.append(len(obs))
        for bi, br in enumerate(brs):
            e = br['e']
            err = br.get('err', np.full(len(e), np.nan))
            conf = feats[tid][br['feat']]['properties']['confidence']
            for k in range(len(e)):
                st_rows.append((i, bi, 0, float(e[k]), float(err[k]), float(br['cov'][k]), br['turn'] * rmin, len(obs),
                                float(br['dco'][k]), float(br['w'][k]), float(br['pk'][k]), float(br['amb'][k]), float(br['fres'][k]),
                                conf, float(br['dot'][k]), k, len(e)))
            ends = []
            for en in br['ends']:
                lg = en['leg']
                d = {'bout': en['bout'], 'partner_line_id': en['partner']['line_id'], 'partner_part_id': en['partner']['part_id'],
                     'partner_src': en['partner']['src'], 'ok': lg is not None}
                if lg is not None:
                    ee = np.sqrt(np.median(e[:10] if en['bout'] == 'debut' else e[-10:]) ** 2 + lg['e_model'] ** 2)
                    errc = lg.get('err') if 'err' in lg else en.get('err')
                    for k in range(len(lg['curve'])):
                        st_rows.append((i, bi, 1, float(ee), float(errc[k]) if errc is not None else np.nan, 1.0, 0.0, len(obs),
                                        np.nan, np.nan, np.nan, 0.0, lg['e_model'], lg['score'], np.nan, k, len(lg['curve'])))
                    d.update({'S': P.to_lonlat(lg['S'][None], s)[0].round(8).tolist(), 's_on_G': lg['s_on_G'],
                              'e_model': round(lg['e_model'], 2), 'score': round(lg['score'], 3), 'n_ok': lg['n_ok'],
                              'err_max': None if errc is None else round(float(np.max(errc)), 2)})
                ends.append(d)
            geom = P.assemble(br)
            bf.write(json.dumps({'tid': tid, 'line_id': t['line_id'], 'part_id': t['part_id'], 'b': bi, 'network': net, 'rmin': rmin,
                                 'coords': P.to_lonlat(geom, s).round(8).tolist(), 'axis_n': len(br['pts']),
                                 'e': np.round(br['e'], 2).tolist(), 'err': None if 'err' not in br else np.round(br['err'], 2).tolist(),
                                 'turn_x_rmin': round(br['turn'] * rmin, 3), 'shift_m': np.round(br['shift_med'], 2).tolist(),
                                 'reg_sd': br['reg_sd'], 'ends': ends}) + '\n')
        if i % 20 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f'{i + 1}/{len(tids)} {tid} branches={len(brs)} obs={len(obs)} rss={rss:.0f} MB', file=sys.stderr, flush=True)
    bf.close()
    arr = np.asarray(st_rows, float).reshape(-1, 17)
    np.savez_compressed(out / f'{tag}-stations.npz', st=arr, tids=np.asarray(tids))
    print(json.dumps({'cibles': len(tids), 'stations': len(arr)}))


if __name__ == '__main__':
    main()
