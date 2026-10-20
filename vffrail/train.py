"""Training of a "railway track axis" segmentation (U-Net, pre-trained ResNet encoder).

Label rasterised on the fly from the ground-truth axes:
  1   = axis ± pos_m / 2
  255 = ignored band (± band_m / 2 around the axis) + the set's "ignored" mask
  0   = negative
A per-chip shift (registration field measured by `offsets`) can be applied
to the axes, to match the map drawing (imperfect georeferencing).

"Relaxed" validation metrics (common in road extraction):
precision = share of the predicted skeleton within tol_m of a true axis;
recall = share of the true axis within tol_m of the predicted skeleton;
offset = median / p90 distance (m) of the predicted skeleton matched to the true axis.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from .geo import draw_lines

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def load_meta(ds: Path):
    return [json.loads(l) for l in open(ds / 'meta.jsonl')]


def build_model(arch='unet', encoder='resnet34', weights='imagenet'):
    import segmentation_models_pytorch as smp

    cls = {'unet': smp.Unet, 'unetpp': smp.UnetPlusPlus, 'fpn': smp.FPN}[arch]
    kw = {'decoder_attention_type': 'scse'} if arch in ('unet', 'unetpp') else {}
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=1, **kw)


def make_label(m, ign, pos_m, band_m, shift=(0.0, 0.0), size=512):
    gres = m['gres']
    lines = [np.asarray(t['pts']) + np.asarray(shift) for t in m['truth']]
    band = draw_lines((size, size), lines, band_m / gres, 1)
    pos = draw_lines((size, size), lines, pos_m / gres, 1)
    center = draw_lines((size, size), lines, 1, 1)
    lab = np.zeros((size, size), np.uint8)
    lab[np.asarray(ign) > 0] = 255
    lab[np.asarray(band) > 0] = 255
    lab[np.asarray(pos) > 0] = 1
    return lab, np.asarray(center) > 0


class ChipDataset(torch.utils.data.Dataset):
    def __init__(self, ds: Path, metas, pos_m, band_m, crop=None, augment=False, shifts=None):
        self.ds, self.metas, self.pos_m, self.band_m = ds, metas, pos_m, band_m
        self.crop, self.augment, self.shifts = crop, augment, shifts or {}

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, i):
        m = self.metas[i]
        ds = Path(m.get('_ds', self.ds))
        img = np.asarray(Image.open(ds / 'img' / f"{m['id']}.png").convert('RGB'), np.float32)
        ign = Image.open(ds / 'ign' / f"{m['id']}.png")
        lab, center = make_label(m, ign, self.pos_m, self.band_m, self.shifts.get(m['id'], (0.0, 0.0)), img.shape[0])
        if self.augment:
            # Hue / contrast: map sheets and photographs vary a lot.
            g = img.mean(axis=2, keepdims=True)
            sat = random.uniform(0.6, 1.3)
            img = g + (img - g) * sat
            img = (img - 128) * random.uniform(0.75, 1.25) + 128 + random.uniform(-25, 25)
            img = img * np.random.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)
            img = np.clip(img, 0, 255)
        if self.crop:
            c = self.crop
            y0 = random.randint(0, img.shape[0] - c)
            x0 = random.randint(0, img.shape[1] - c)
            img, lab, center = img[y0:y0 + c, x0:x0 + c], lab[y0:y0 + c, x0:x0 + c], center[y0:y0 + c, x0:x0 + c]
        if self.augment:
            k = random.randint(0, 3)
            img, lab, center = np.rot90(img, k), np.rot90(lab, k), np.rot90(center, k)
            if random.random() < 0.5:
                img, lab, center = img[:, ::-1], lab[:, ::-1], center[:, ::-1]
        # uint8: 4 × less memory in the DataLoader queues; normalisation on the GPU.
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)).round().astype(np.uint8))
        return x, torch.from_numpy(lab.copy()), torch.from_numpy(center.copy())


def masked_loss(logits, lab, pos_weight=2.0):
    valid = (lab != 255).float()
    tgt = (lab == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none',
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    inter = (p * tgt).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (tgt * valid).sum() + 1)
    return bce + dice


def relaxed_metrics(prob, lab, center, gres, tol_m=10.0, thr=0.5):
    """prob, lab, center: 2D numpy arrays of a chip. Returns counters."""
    pred = skeletonize(prob > thr)
    valid = lab != 255
    tol = tol_m / gres
    out = {'pred': 0, 'pred_tp': 0, 'true': 0, 'true_tp': 0, 'dists': []}
    if center.any():
        d_true = ndimage.distance_transform_edt(~center)
        # Counted: outside the "ignored" mask, or within 3 × tol of the true axis (the training
        # ignored band around the axis must not hide the true positives).
        pv = pred & (valid | (d_true <= 3 * tol))
        pred_v = pv
        out['pred'] = int(pv.sum())
        out['pred_tp'] = int((pv & (d_true <= tol)).sum())
        dd = d_true[pv & (d_true <= 3 * tol)] * gres
        out['dists'] = dd.tolist()
    else:
        out['pred'] = int((pred & valid).sum())
    if center.any():
        out['true'] = int(center.sum())
        if pred.any():
            d_pred = ndimage.distance_transform_edt(~pred)
            out['true_tp'] = int((center & (d_pred <= tol)).sum())
    return out


def summarize(counts):
    P = sum(c['pred'] for c in counts)
    TPp = sum(c['pred_tp'] for c in counts)
    T = sum(c['true'] for c in counts)
    TPt = sum(c['true_tp'] for c in counts)
    d = np.concatenate([np.asarray(c['dists']) for c in counts if c['dists']] or [np.zeros(0)])
    prec = TPp / P if P else 0.0
    rec = TPt / T if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'precision': round(prec, 4), 'rappel': round(rec, 4), 'f1': round(f1, 4),
            'ecart_median_m': round(float(np.median(d)), 2) if len(d) else None,
            'ecart_p90_m': round(float(np.percentile(d, 90)), 2) if len(d) else None}


def normalize(x, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return ((x.to(device, non_blocking=True).float() - mean) / std).contiguous(memory_format=torch.channels_last)


def evaluate(model, loader, metas, device, tol_m, thr=0.5):
    model.eval()
    counts, k = [], 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
        for x, lab, center in loader:
            prob = torch.sigmoid(model(normalize(x, device))).float().cpu().numpy()[:, 0]
            for b in range(prob.shape[0]):
                counts.append(relaxed_metrics(prob[b], lab[b].numpy(), center[b].numpy(), metas[k]['gres'], tol_m, thr))
                k += 1
    return summarize(counts)


def _worker_init(_):
    torch.set_num_threads(1)


def run(args, log=print):
    from .throttle import GpuThrottle, limit_memory

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    limit_memory(args.gpu_max_mem_gb)
    throttle = GpuThrottle(args.gpu_max_util, log=log)
    log(f'device: {device} {torch.cuda.get_device_name(0) if device.type == "cuda" else ""}')
    # One or more sets (comma-separated): multi-region training.
    names = args.dataset.split(',')
    ds = Path(args.work) / 'datasets' / names[0]
    metas = []
    for name in names:
        d = Path(args.work) / 'datasets' / name
        for m in load_meta(d):
            m['_ds'] = str(d)
            metas.append(m)
    shifts = {}
    if args.shifts:
        shifts = {k: tuple(v) for k, v in json.load(open(args.shifts)).items()}
        log(f'  shifts applied to the axes: {len(shifts)} chips')
    train_m = [m for m in metas if m['fold'] == 'train']
    val_m = [m for m in metas if m['fold'] == 'val' and m['truth']]
    log(f'  {len(train_m)} training chips, {len(val_m)} validation chips')

    tr = ChipDataset(ds, train_m, args.pos_m, args.band_m, crop=args.crop, augment=True, shifts=shifts)
    va = ChipDataset(ds, val_m, args.pos_m, args.band_m, shifts=shifts)
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                     pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=2,
                                     worker_init_fn=_worker_init)
    vl = torch.utils.data.DataLoader(va, batch_size=8, shuffle=False, num_workers=2, worker_init_fn=_worker_init)
    model = build_model(args.arch, args.encoder, None if args.init else 'imagenet').to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)['state'])
        log(f'  initialised from {args.init}')
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    out = Path(args.work) / 'models' / args.out
    out.mkdir(parents=True, exist_ok=True)
    best, hist = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, lab, _ in tl:
            throttle.start()
            x = normalize(x, device)
            lab = lab.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                logits = model(x)[:, 0]
            loss = masked_loss(logits.float(), lab, args.pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            throttle.end()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        dt = time.time() - t0
        rec = {'epoch': ep + 1, 'loss': round(tot / max(n, 1), 4), 'img_s': round(n / dt, 1)}
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.epochs:
            rec.update(evaluate(model, vl, val_m, device, args.tol_m))
            if rec['f1'] > best:
                best = rec['f1']
                torch.save({'state': model.state_dict(), 'arch': args.arch, 'encoder': args.encoder,
                            'dataset': args.dataset, 'pos_m': args.pos_m, 'band_m': args.band_m,
                            'metrics': rec, 'shifts': bool(args.shifts)}, out / 'best.pt')
        hist.append(rec)
        log(json.dumps(rec, ensure_ascii=False))
        if (ep + 1) % 5 == 0:
            log('  ' + throttle.summary())
    torch.save({'state': model.state_dict(), 'arch': args.arch, 'encoder': args.encoder, 'dataset': args.dataset,
                'pos_m': args.pos_m, 'band_m': args.band_m, 'metrics': hist[-1]}, out / 'last.pt')
    json.dump(hist, open(out / 'history.json', 'w'), indent=1)
    log(f'done: best validation F1 {best:.4f} -> {out / "best.pt"}')


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    model = build_model(ck['arch'], ck['encoder'], None)
    model.load_state_dict(ck['state'])
    return model.to(device).eval().to(memory_format=torch.channels_last), ck


def predict_batch(model, imgs: np.ndarray, device, tta=True):
    """imgs: (B, H, W, 3) uint8 -> probabilities (B, H, W) float32. TTA: 4 rotations."""
    x = torch.from_numpy(((imgs.astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2).copy()).to(device)
    x = x.to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
        if not tta:
            return torch.sigmoid(model(x)).float()[:, 0].cpu().numpy()
        acc = None
        for k in range(4):
            p = torch.sigmoid(model(torch.rot90(x, k, (2, 3)))).float()
            p = torch.rot90(p, -k, (2, 3))
            acc = p if acc is None else acc + p
        return (acc / 4)[:, 0].cpu().numpy()
