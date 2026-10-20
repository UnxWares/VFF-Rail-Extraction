"""Measurement of the georeferencing offset of the basemap (map or photograph) against the truth.

1. Model (v0) prediction on each chip of the set.
2. For each point of the true axes: position of the probability peak along the
   normal (± search_m) -> observation (point, normal, signed offset in m).
3. Smoothed registration field (field.OffsetField) -> shift per chip (JSON, px)
   to relabel (v1), and statistics: raw offset, systematic part
   (predictable from the neighbouring tracks) and residual.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .field import OffsetField, normal_profiles
from .geo import TILE, merc_scale, px_to_merc
from .train import load_meta, load_model, predict_batch

CHIP = 2 * TILE


def chip_shift(img, prob, truth_lines, max_px=15, margin=16):
    """Shift (dx, dy) in px of the drawing relative to the true axes of a chip.

    Response R = probability (wide-band model) × darkness of the stroke; score of a
    shift = mean of R along the shifted true axes. Returns
    (dx, dy, confidence, anisotropy, normal (nx, ny)) or None.
    """
    from scipy.ndimage import map_coordinates

    H, W = prob.shape
    gray = img.astype(np.float32).mean(axis=2)
    R = prob * np.clip((200.0 - gray) / 200.0, 0, 1)
    pts = []
    for ls in truth_lines:
        a = np.asarray(ls, float)
        if len(a) < 2:
            continue
        seg = np.diff(a, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0], np.cumsum(L)])
        s = np.arange(0, cum[-1], 1.0)
        p = np.stack([np.interp(s, cum, a[:, 0]), np.interp(s, cum, a[:, 1])], 1)
        ok = (p[:, 0] > margin + max_px) & (p[:, 0] < W - margin - max_px) & (p[:, 1] > margin + max_px) & (p[:, 1] < H - margin - max_px)
        pts.append(p[ok])
    if not pts:
        return None
    P = np.concatenate(pts)
    if len(P) < 80:
        return None
    shifts = np.arange(-max_px, max_px + 1)
    S = np.zeros((len(shifts), len(shifts)))
    C0 = np.cov((P - P.mean(0)).T)
    ev0 = np.linalg.eigvalsh(C0)
    if ev0[1] / max(ev0[0], 1e-6) <= 25:
        for j, dy in enumerate(shifts):
            for i, dx in enumerate(shifts):
                S[j, i] = map_coordinates(R, [P[:, 1] + dy, P[:, 0] + dx], order=1).mean()
    def para(a, b, c):
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if den < 0 else 0.0

    # Anisotropy (almost straight axes): the 2D score forms a ridge along the track,
    # only the normal component is measurable -> 1D profile along the mean normal.
    C = np.cov((P - P.mean(0)).T)
    ev, evec = np.linalg.eigh(C)
    aniso = ev[1] / max(ev[0], 1e-6)
    n = evec[:, 0]  # direction of smallest extent = mean normal
    if aniso > 25:
        ds = np.arange(-max_px, max_px + 0.01, 0.5)
        prof = np.array([map_coordinates(R, [P[:, 1] + d * n[1], P[:, 0] + d * n[0]], order=1).mean() for d in ds])
        k = int(prof.argmax())
        best = prof[k]
        side = prof[np.abs(ds - ds[k]) > 6]  # secondary lobe: more than ~20 m from the peak
        conf = (best - (side.max() if len(side) else 0.0)) / max(best, 1e-6)
        d = ds[k] + (0.5 * para(prof[k - 1], prof[k], prof[k + 1]) if 0 < k < len(ds) - 1 else 0.0)
        dx, dy = d * n[0], d * n[1]
        edge = abs(ds[k]) >= max_px - 1
    else:
        j, i = np.unravel_index(S.argmax(), S.shape)
        best = S[j, i]
        yy, xx = np.meshgrid(shifts, shifts, indexing='ij')
        far = np.hypot(yy - shifts[j], xx - shifts[i]) > 6
        conf = (best - S[far].max()) / max(best, 1e-6)
        dx = shifts[i] + (para(S[j, i - 1], S[j, i], S[j, i + 1]) if 0 < i < len(shifts) - 1 else 0.0)
        dy = shifts[j] + (para(S[j - 1, i], S[j, i], S[j + 1, i]) if 0 < j < len(shifts) - 1 else 0.0)
        edge = abs(shifts[i]) >= max_px - 1 or abs(shifts[j]) >= max_px - 1
    return float(dx), float(dy), float(conf), float(aniso), (float(n[0]), float(n[1])), bool(edge), float(best)


def run_corr(args, log=print):
    """Shift per chip by correlation (method retained for the SCAN 50)."""
    from .throttle import GpuThrottle, limit_memory

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    limit_memory(args.gpu_max_mem_gb)
    throttle = GpuThrottle(args.gpu_max_util, log=log)
    ds = Path(args.work) / 'datasets' / args.dataset
    mdir = Path(args.work) / 'models' / args.model
    model, ck = load_model(mdir / 'best.pt', device)
    metas = [m for m in load_meta(ds) if m['truth']]
    max_px = int(round(args.search_m / 3.2))
    res, obs = {}, []
    for i in range(0, len(metas), args.batch):
        batch = metas[i:i + args.batch]
        imgs = np.stack([np.asarray(Image.open(ds / 'img' / f"{m['id']}.png").convert('RGB')) for m in batch])
        pc = Path(args.work) / 'prob-chips' / args.model / args.dataset
        pc.mkdir(parents=True, exist_ok=True)
        cached = [pc / f"{m['id']}.png" for m in batch]
        if all(c.exists() for c in cached):
            probs = np.stack([np.asarray(Image.open(c), np.float32) / 255 for c in cached])
        else:
            throttle.start()
            probs = predict_batch(model, imgs, device, tta=True)
            throttle.end()
            for c, pr in zip(cached, probs):
                Image.fromarray((pr * 255).round().astype(np.uint8)).save(c, compress_level=1)
        for m, img, prob in zip(batch, imgs, probs):
            r = chip_shift(img, prob, [t['pts'] for t in m['truth']], max_px=max_px)
            if r is None:
                continue
            dx, dy, conf, aniso, n, edge, best = r
            res[m['id']] = {'dx': round(dx, 2), 'dy': round(dy, 2), 'conf': round(conf, 2), 'aniso': round(min(aniso, 999), 1),
                            'edge': edge, 'score': round(best, 4), 'fold': m['fold'], 'gres': m['gres'], 'lat': m['lat']}
            if conf >= args.min_conf and best >= 0.2 and not edge:
                # "Normal" observations for the registration field (ground frame east / north, m).
                x0, y0 = m['bx'] * CHIP, m['by'] * CHIP
                for t in m['truth']:
                    a = np.asarray(t['pts'], float)
                    for k in range(0, len(a) - 1):
                        mid = (a[k] + a[k + 1]) / 2
                        if not (0 <= mid[0] < CHIP and 0 <= mid[1] < CHIP):
                            continue
                        tv = a[k + 1] - a[k]
                        L = np.hypot(*tv)
                        if L < 4:
                            continue
                        nx, ny = -tv[1] / L, tv[0] / L
                        d = (dx * nx + dy * ny) * m['gres']
                        mx, my = px_to_merc(mid[0] + x0, mid[1] + y0, m['z'])
                        obs.append((mx, my, nx, -ny, d, conf, t['line_id'], 1 if m['fold'] == 'val' else 0, m['lat']))
        if (i // args.batch) % 25 == 0:
            log(f'  {i + len(batch)}/{len(metas)} chips')
    json.dump(res, open(mdir / f'corr-{args.dataset}.json', 'w'))
    obs = np.asarray(obs, float).reshape(-1, 9)
    np.save(mdir / f'offsets-obs-{args.dataset}.npy', obs)
    good = {k: v for k, v in res.items() if v['conf'] >= args.min_conf and v['score'] >= 0.2 and not v['edge']}
    mags = np.asarray([np.hypot(v['dx'], v['dy']) * v['gres'] for v in good.values()])
    stats = {'vignettes': len(res), 'vignettes_fiables': len(good),
             'decalage_median_m': round(float(np.median(mags)), 1) if len(mags) else None,
             'decalage_p90_m': round(float(np.percentile(mags, 90)), 1) if len(mags) else None,
             'observations': int(len(obs))}
    # Part predictable from the neighbours (leave-one-line-out): does the field explain the shift?
    if len(obs):
        field = OffsetField(obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], w=obs[:, 5], group=obs[:, 6])
        rng = np.random.default_rng(0)
        idx = rng.choice(len(obs), size=min(3000, len(obs)), replace=False)
        raw, cor = [], []
        for k in idx:
            f = field.at(obs[k, 0], obs[k, 1], merc_scale(obs[k, 8]), exclude_group=obs[k, 6], min_obs=20)
            raw.append(abs(obs[k, 4]))
            cor.append(abs(obs[k, 4] - (f[0] * obs[k, 2] + f[1] * obs[k, 3])) if f else abs(obs[k, 4]))
        stats.update({'normale_brute_median_m': round(float(np.median(raw)), 1), 'normale_brute_p90_m': round(float(np.percentile(raw, 90)), 1),
                      'normale_apres_champ_median_m': round(float(np.median(cor)), 1),
                      'normale_apres_champ_p90_m': round(float(np.percentile(cor, 90)), 1)})
    # Shifts applied to the labels: reliable chip -> its shift; otherwise none (wide band).
    shifts = {k: [v['dx'], v['dy']] for k, v in good.items()}
    json.dump(shifts, open(mdir / f'shifts-{args.dataset}.json', 'w'))
    json.dump(stats, open(mdir / f'offsets-{args.dataset}.json', 'w'), indent=1, ensure_ascii=False)
    log(json.dumps(stats, ensure_ascii=False))
    log(f'done: {len(shifts)} chips registered -> {mdir / f"shifts-{args.dataset}.json"}')


def run(args, log=print):
    if getattr(args, 'method', 'corr') == 'corr':
        return run_corr(args, log)
    from .throttle import GpuThrottle, limit_memory

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    limit_memory(args.gpu_max_mem_gb)
    throttle = GpuThrottle(args.gpu_max_util, log=log)
    ds = Path(args.work) / 'datasets' / args.dataset
    mdir = Path(args.work) / 'models' / args.model
    model, ck = load_model(mdir / 'best.pt', device)
    metas = [m for m in load_meta(ds) if m['truth']]
    log(f'{len(metas)} chips with ground truth, model {args.model} ({ck["metrics"]})')
    obs = []
    for i in range(0, len(metas), args.batch):
        batch = metas[i:i + args.batch]
        imgs = np.stack([np.asarray(Image.open(ds / 'img' / f"{m['id']}.png").convert('RGB')) for m in batch])
        throttle.start()
        probs = predict_batch(model, imgs, device, tta=True)
        throttle.end()
        for m, prob in zip(batch, probs):
            gres = m['gres']
            for t in m['truth']:
                o = normal_profiles(prob, [np.asarray(t['pts'])], args.search_m / gres)
                if not len(o):
                    continue
                x0, y0 = m['bx'] * CHIP, m['by'] * CHIP
                mx, my = px_to_merc(o[:, 0] + x0, o[:, 1] + y0, m['z'])
                for k in range(len(o)):
                    # ground frame: east, north (the pixel row axis points south)
                    obs.append((mx[k], my[k], o[k, 2], -o[k, 3], o[k, 4] * gres, o[k, 5], t['line_id'],
                                1 if m['fold'] == 'val' else 0, m['lat']))
        if (i // args.batch) % 20 == 0:
            log(f'  {i + len(batch)}/{len(metas)} chips, {len(obs)} observations')
    obs = np.asarray(obs, float)
    np.save(mdir / f'offsets-obs-{args.dataset}.npy', obs)
    log(f'{len(obs)} observations')

    d = obs[:, 4]
    log(f'raw offset along the normal: median |d| {np.median(np.abs(d)):.1f} m, p90 {np.percentile(np.abs(d), 90):.1f} m')

    # Field estimated WITHOUT the evaluated line (leave-one-line-out): predictable part of the shift.
    field = OffsetField(obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], obs[:, 4], w=obs[:, 5], group=obs[:, 6])
    rng = np.random.default_rng(0)
    sample = rng.choice(len(obs), size=min(4000, len(obs)), replace=False)
    resid_raw, resid_corr, found = [], [], 0
    for k in sample:
        f = field.at(obs[k, 0], obs[k, 1], merc_scale(obs[k, 8]), exclude_group=obs[k, 6])
        resid_raw.append(abs(obs[k, 4]))
        if f is None:
            resid_corr.append(abs(obs[k, 4]))
            continue
        found += 1
        pred = f[0] * obs[k, 2] + f[1] * obs[k, 3]
        resid_corr.append(abs(obs[k, 4] - pred))
    resid_raw, resid_corr = np.asarray(resid_raw), np.asarray(resid_corr)
    stats = {
        'observations': int(len(obs)),
        'ecart_brut_median_m': round(float(np.median(resid_raw)), 2),
        'ecart_brut_p90_m': round(float(np.percentile(resid_raw, 90)), 2),
        'ecart_apres_recalage_median_m': round(float(np.median(resid_corr)), 2),
        'ecart_apres_recalage_p90_m': round(float(np.percentile(resid_corr, 90)), 2),
        'points_recalables_pct': round(100 * found / len(sample), 1),
    }
    log(json.dumps(stats, ensure_ascii=False))

    # Shift per chip (all points, full field) -> v1 labels.
    shifts, mags = {}, []
    for m in load_meta(ds):
        x0, y0 = m['bx'] * CHIP + CHIP / 2, m['by'] * CHIP + CHIP / 2
        mx, my = px_to_merc(x0, y0, m['z'])
        f = field.at(float(mx), float(my), merc_scale(m['lat']))
        if f is None:
            continue
        dx, dy = f[0], f[1]
        mags.append(np.hypot(dx, dy))
        shifts[m['id']] = [round(dx / m['gres'], 2), round(-dy / m['gres'], 2)]
    json.dump(shifts, open(mdir / f'shifts-{args.dataset}.json', 'w'))
    stats['vignettes_recalees'] = len(shifts)
    if mags:
        stats['decalage_champ_median_m'] = round(float(np.median(mags)), 2)
        stats['decalage_champ_p90_m'] = round(float(np.percentile(mags, 90)), 2)
    json.dump(stats, open(mdir / f'offsets-{args.dataset}.json', 'w'), indent=1, ensure_ascii=False)
    log(f'done: {len(shifts)} chips registered -> {mdir / f"shifts-{args.dataset}.json"}')
