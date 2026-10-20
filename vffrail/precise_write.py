"""Strict filter and preparation of the writing (DRY-RUN only) of the precise junction branches.

    py.sh -m vffrail.precise_write --stations X-stations.npz --branches X-branches.jsonl --out DIR [--write]

Nothing is written to the database without --write, and --write is refused in this version (dry-run only).

Per axis station: "precise" if the offset from the coarse trace <= FILTER['dco'], observation coverage >= FILTER['cov'],
path confidence >= FILTER['conf'], plateau <= FILTER['w'] (thresholds calibrated on half of the control corridors,
validated on the other half). A switch curve is "precise" if e_model <= LEG_E_MAX and the LEG_AXIS_M m of axis that
extend it are precise. The rest is "estimated". The geometry written is CONTINUOUS: precise and estimated pieces share
their vertices, and each end that joins a partner track ends EXACTLY on a point of its geometry (S), to be inserted
as a vertex in the partner track (properties.aiguillages[].S, partner_part_id).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FILTER = {'dco': 3.0, 'cov': 0.9, 'conf': 0.85, 'w': 22.0}
LEG_E_MAX = 2.0
LEG_AXIS_M = 30.0
MIN_PRECISE_M = 50.0
STEP_M = 2.0


def axis_precise(rows: np.ndarray) -> np.ndarray:
    """rows: axis stations (precise_run columns). Mask of the precise stations."""
    return ((rows[:, 8] <= FILTER['dco']) & (rows[:, 5] >= FILTER['cov']) & (rows[:, 13] >= FILTER['conf'])
            & (np.nan_to_num(rows[:, 9], nan=99.0) <= FILTER['w']))


def _min_runs(mask, min_n):
    m = mask.copy()
    idx = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    for a, b in zip(idx[::2], idx[1::2]):
        if b - a < min_n:
            m[a:b] = False
    return m


def branch_labels(st_b: np.ndarray, ends: list[dict]):
    """Labels (True = precise) of the vertices of the assembled geometry [head, axis, tail] of a branch."""
    ax = st_b[st_b[:, 2] == 0]
    ax = ax[np.argsort(ax[:, 15])]
    lab_ax = _min_runs(axis_precise(ax), int(MIN_PRECISE_M / STEP_M))
    legs = st_b[st_b[:, 2] == 1]
    head = tail = None
    pos = 0
    nax = int(LEG_AXIS_M / STEP_M)
    for en in ends:
        if not en['ok']:
            continue
        n = int(legs[pos, 16]) if pos < len(legs) else 0
        lg = legs[pos:pos + n]
        pos += n
        e_model = float(lg[0, 12]) if n else 99.0
        if en['bout'] == 'debut':
            ok = e_model <= LEG_E_MAX and lab_ax[:nax].all()
            head = np.full(n - 1, ok)
        else:
            ok = e_model <= LEG_E_MAX and lab_ax[-nax:].all()
            tail = np.full(n - 1, ok)
    parts = [x for x in (head, lab_ax, tail) if x is not None]
    return np.concatenate(parts)


def runs_of(coords, lab):
    """Splits a polyline into pieces of the same label that share their junction vertices."""
    out = []
    a = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[a]:
            seg = coords[max(0, a - 1):i]  # each piece starts at the last vertex of the previous one
            out.append((bool(lab[a]), seg))
            a = i
    return out


def length_m(ll):
    ll = np.asarray(ll)
    if len(ll) < 2:
        return 0.0
    lat = np.radians(ll[:, 1].mean())
    d = np.diff(ll, axis=0) * np.array([np.cos(lat), 1.0]) * 111_195.0
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--stations', required=True)
    ap.add_argument('--branches', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--eras', action='store_true', help='km per era (read-only access to line_segments)')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args(argv)
    if a.write:
        raise SystemExit("writing is not allowed in this version: dry-run only")
    st = np.load(a.stations)['st']
    brs = [json.loads(x) for x in open(a.branches)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tids = list(np.load(a.stations)['tids'])
    tix = {t: i for i, t in enumerate(tids)}
    key = st[:, 0] * 1000 + st[:, 1]
    feats, km = [], {}
    n_join, n_join_ok = 0, 0
    for b in brs:
        sb = st[key == tix[b['tid']] * 1000 + b['b']]
        lab = branch_labels(sb, b['ends'])
        coords = np.asarray(b['coords'])
        if len(lab) != len(coords):
            raise SystemExit(f"{b['tid']} b{b['b']}: {len(lab)} labels for {len(coords)} vertices")
        for en in b['ends']:
            n_join += 1
            n_join_ok += int(en['ok'])
        for prec, seg in runs_of(coords, lab):
            L = length_m(seg)
            if L <= 0:
                continue
            k = (b['network'] or '?', 'precis' if prec else 'estime')
            km.setdefault(b['line_id'], {}).setdefault(k, 0.0)
            km[b['line_id']][k] += L / 1000
            feats.append({'type': 'Feature', 'geometry': {'type': 'LineString', 'coordinates': np.round(seg, 8).tolist()},
                          'properties': {'line_id': b['line_id'], 'part_id': b['part_id'], 'tid': b['tid'], 'precision': 'precis' if prec else 'estime',
                                         'len_m': round(L, 1), 'network': b['network'],
                                         'aiguillages': [e for e in b['ends'] if e['ok']]}})
    json.dump({'type': 'FeatureCollection', 'features': feats}, open(out / 'write-preview.geojson', 'w'))
    by_net = {}
    for lid, d in km.items():
        for (net, cl), v in d.items():
            by_net.setdefault(net, {}).setdefault(cl, 0.0)
            by_net[net][cl] += v
    rep = {'filtre': FILTER, 'aiguillage_e_max_m': LEG_E_MAX, 'morceaux': len(feats),
           'bouts_jonction': n_join, 'aiguillages_construits': n_join_ok,
           'km_par_reseau': {n: {c: round(v, 2) for c, v in d.items()} for n, d in sorted(by_net.items())},
           'km_total': {c: round(sum(d.get(c, 0.0) for d in by_net.values()), 2) for c in ('precis', 'estime')}}
    if a.eras:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from vffrail import db as vdb

        conn = vdb.connect()
        rows = conn.execute("""SELECT DISTINCT e.year, ls.line_id FROM line_segments ls JOIN eras e ON e.id = ls.era_id
                               WHERE ls.line_id = ANY(%s)""", (sorted(km),)).fetchall()
        era = {}
        for y, lid in rows:
            for (net, cl), v in km[int(lid)].items():
                era.setdefault(int(y), {}).setdefault(cl, 0.0)
                era[int(y)][cl] += v
        rep['km_par_ere'] = {y: {c: round(v, 2) for c, v in d.items()} for y, d in sorted(era.items())}
        rep['note_eres'] = 'km of the branches of the lines present in the era (line_segments), the whole branch counted'
    json.dump(rep, open(out / 'dry-run.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == '__main__':
    main()
