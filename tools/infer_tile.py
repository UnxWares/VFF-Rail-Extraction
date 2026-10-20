"""Standalone inference: RGB image (aerial photo, z16, ~1.6 m/px) -> railway-axis probability PNG.

    ./run.sh python -I /app/tools/infer_tile.py /app/models/ortho-v1/ortho-v1.fp16.safetensors in.png out.png [--tta]

Needs no database. Input: any RGB image (PNG/JPEG); it is padded to a multiple of 32 px.
Output: 8-bit grayscale PNG, 255 = probability 1 that the pixel lies within +/- 3 m of a track axis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from vffrail.train import MEAN, STD, build_model  # noqa: E402


def load(path: str, device) -> torch.nn.Module:
    net = build_model('unet', 'resnet34', None)
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in load_file(path).items()}
    net.load_state_dict(sd)
    return net.to(device).eval()


def predict(net, rgb: np.ndarray, device, tta=False) -> np.ndarray:
    h, w = rgb.shape[:2]
    H, W = -(-h // 32) * 32, -(-w // 32) * 32
    pad = np.pad(rgb, ((0, H - h), (0, W - w), (0, 0)), mode='reflect')
    x = torch.from_numpy(((pad.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None].copy()).to(device)
    with torch.no_grad():
        if not tta:
            p = torch.sigmoid(net(x))
        else:
            p = sum(torch.rot90(torch.sigmoid(net(torch.rot90(x, k, (2, 3)))), -k, (2, 3)) for k in range(4)) / 4
    return p[0, 0, :h, :w].cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('weights')
    ap.add_argument('image')
    ap.add_argument('out')
    ap.add_argument('--tta', action='store_true', help='average over 4 rotations')
    a = ap.parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load(a.weights, device)
    rgb = np.asarray(Image.open(a.image).convert('RGB'))
    prob = predict(net, rgb, device, a.tta)
    Image.fromarray(np.round(prob * 255).astype(np.uint8)).save(a.out)
    print(f'{a.out}: max {prob.max():.3f}, pixels > 0.5: {(prob > 0.5).mean() * 100:.2f} %')


if __name__ == '__main__':
    main()
