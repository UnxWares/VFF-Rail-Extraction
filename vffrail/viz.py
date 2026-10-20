"""Check previews (PNG): chip + truth / ignored mask / probability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def chip_overlay(ds: Path, chip_id: str, out: Path, prob: np.ndarray | None = None, shift=(0.0, 0.0)):
    m = next(json.loads(l) for l in open(ds / 'meta.jsonl') if json.loads(l)['id'] == chip_id)
    img = Image.open(ds / 'img' / f'{chip_id}.png').convert('RGB')
    ign = np.asarray(Image.open(ds / 'ign' / f'{chip_id}.png')) > 0
    a = np.asarray(img).astype(np.float32)
    a[ign] = a[ign] * 0.6 + np.array([255, 255, 0]) * 0.4
    if prob is not None:
        a = a * (1 - prob[..., None] * 0.7) + np.array([255, 0, 255]) * prob[..., None] * 0.7
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for t in m['truth']:
        pts = [(x + shift[0], y + shift[1]) for x, y in t['pts']]
        d.line(pts, fill=(255, 0, 0), width=1)
    im.save(out)
    return m
