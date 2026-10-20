"""Evaluation on chips (pixel F1) and exports for publication (safetensors, ONNX, SHA-256).

    rail.py evalchips --model ortho-v1 --dataset ctl-ortho-1950-alpes[,…]   # never-seen chips
    rail.py export --model ortho-v1 --out /work/release/ortho-v1 --check-dataset ctl-…

The ONNX model includes the normalisation and the sigmoid: float32 input (N, 3, H, W) in
RGB 0-255 values (H, W multiples of 32), output (N, 1, H, W) = "track axis" probability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .train import MEAN, STD, ChipDataset, build_model, evaluate, load_meta, load_model


class WithPrePost(torch.nn.Module):
    """Published model: RGB 0-255 -> probability (ImageNet normalisation and sigmoid included)."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.sigmoid(self.net((x - self.mean) / self.std))


def _chips_eval(model, names, work, device, pos_m, band_m, tol_m, log):
    out = {}
    for name in names:
        d = Path(work) / 'datasets' / name
        metas = [m for m in load_meta(d) if m['truth']]
        dsx = ChipDataset(d, metas, pos_m, band_m)
        dl = torch.utils.data.DataLoader(dsx, batch_size=8, num_workers=2)
        out[name] = {'vignettes': len(metas), **evaluate(model, dl, metas, device, tol_m)}
        log(f'  {name}: {json.dumps(out[name], ensure_ascii=False)}')
    return out


def evalchips(args, log=print):
    from .throttle import limit_memory

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    limit_memory(args.gpu_max_mem_gb)
    model, ck = load_model(Path(args.work) / 'models' / args.model / 'best.pt', device)
    res = _chips_eval(model, args.dataset.split(','), args.work, device, ck.get('pos_m', 6), ck.get('band_m', 14), args.tol_m, log)
    out = Path(args.work) / 'models' / args.model / 'evalchips.json'
    prev = json.load(open(out)) if out.exists() else {}
    prev.update(res)
    json.dump(prev, open(out, 'w'), indent=1, ensure_ascii=False)
    log(f'done -> {out}')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def export(args, log=print):
    from safetensors.torch import load_file, save_file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src = Path(args.work) / 'models' / args.model / 'best.pt'
    ck = torch.load(src, map_location='cpu')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {'arch': ck['arch'], 'encoder': ck['encoder'], 'classes': '1', 'input': 'RGB 0-255, normalisation ImageNet',
            'mean': json.dumps(MEAN.tolist()), 'std': json.dumps(STD.tolist()), 'layer': 'ORTHOIMAGERY.ORTHOPHOTOS.1950-1965',
            'zoom': '16 (Web Mercator, ~1,6 m/px)', 'dataset': ck.get('dataset', ''), 'metrics': json.dumps(ck.get('metrics', {}))}
    state = {k: v.contiguous() for k, v in ck['state'].items()}
    f32 = out / f'{args.model}.fp32.safetensors'
    f16 = out / f'{args.model}.fp16.safetensors'
    save_file(state, str(f32), metadata=meta)
    save_file({k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}, str(f16), metadata=meta)

    # fp16 check: same metrics as fp32 on control chips?
    checks = {}
    if args.check_dataset:
        names = args.check_dataset.split(',')
        for tag, path in (('fp32', f32), ('fp16', f16)):
            net = build_model(ck['arch'], ck['encoder'], None)
            sd = {k: v.float() if v.is_floating_point() else v for k, v in load_file(str(path)).items()}
            net.load_state_dict(sd)
            net = net.to(device).eval().to(memory_format=torch.channels_last)
            checks[tag] = _chips_eval(net, names, args.work, device, ck.get('pos_m', 6), ck.get('band_m', 14), args.tol_m, log)

    onnx_diff = None
    if not getattr(args, 'no_onnx', False):
        # ONNX (normalisation + sigmoid included, dynamic H and W).
        net = build_model(ck['arch'], ck['encoder'], None)
        net.load_state_dict(state)
        wrapped = WithPrePost(net).eval()
        onnx_path = out / f'{args.model}.onnx'
        dummy = torch.rand(1, 3, 512, 512) * 255
        torch.onnx.export(wrapped, dummy, str(onnx_path), input_names=['rgb'], output_names=['prob'], opset_version=17,
                          dynamic_axes={'rgb': {0: 'n', 2: 'h', 3: 'w'}, 'prob': {0: 'n', 2: 'h', 3: 'w'}}, dynamo=False)
        # ONNX Runtime vs PyTorch check.
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        x = (torch.rand(2, 3, 384, 512) * 255)
        with torch.no_grad():
            ref = wrapped(x).numpy()
        got = sess.run(None, {'rgb': x.numpy()})[0]
        onnx_diff = float(np.abs(ref - got).max())
        log(f'  ONNX: max difference to PyTorch {onnx_diff:.2e}')

    sums = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.suffix in ('.safetensors', '.onnx')}
    with open(out / 'SHA256SUMS', 'w') as f:
        for name, h in sums.items():
            f.write(f'{h}  {name}\n')
    info = {'source': f'{args.model}/best.pt', 'files': {n: {'sha256': h, 'bytes': (out / n).stat().st_size} for n, h in sums.items()},
            'onnx_max_abs_diff': onnx_diff, 'fp16_vs_fp32': checks, 'metadata': meta}
    json.dump(info, open(out / 'export.json', 'w'), indent=1, ensure_ascii=False)
    log(f'done: exports -> {out} ({", ".join(f"{n} {(out / n).stat().st_size / 1e6:.1f} MB" for n in sums)})')
