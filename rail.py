"""Railway track extraction from old IGN maps and photographs (VFF).

    python -I /app/rail.py dataset  --layer scan50-1950 --region bretagne --eras 1940,1950
    python -I /app/rail.py train    --dataset scan50-1950-bretagne --out scan50-v0
    python -I /app/rail.py offsets  --dataset scan50-1950-bretagne --model scan50-v0
    python -I /app/rail.py extract  --layer scan50-1950 --model scan50-v1 --region bretagne
    python -I /app/rail.py evaluate --layer scan50-1950 --model scan50-v1 --region bretagne
    python -I /app/rail.py write    --run <output folder>   # -> source_traces (source = 'ign-historique')

Launched through ./run.sh (container, GPU, host network).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# -I: the script folder is not in sys.path; add the code folder (/app) explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

REGIONS = {
    # lon0, lat0, lon1, lat1
    'bretagne': (-5.2, 47.25, -1.0, 48.95),
    'normandie': (-1.95, 48.15, 1.8, 50.1),
    'nord': (1.5, 49.6, 4.3, 51.1),
    'centre': (0.0, 46.3, 3.2, 48.6),
    'alsace-lorraine': (5.4, 47.4, 8.3, 49.6),
    'sud-ouest': (-1.8, 42.7, 2.0, 45.6),
    'sud-est': (3.6, 42.9, 7.8, 45.3),
    'massif-central': (1.4, 44.1, 4.6, 46.5),
    'france': (-5.3, 41.3, 9.7, 51.2),
}


def bbox_arg(a):
    if a.region:
        return REGIONS[a.region]
    if a.bbox:
        return tuple(float(v) for v in a.bbox.split(','))
    return REGIONS['france']


def main():
    sys.stdout.reconfigure(line_buffering=True)  # -I ignores PYTHONUNBUFFERED: live logs
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--work', default='/work')
    p.add_argument('--tiles', default='/data')
    sub = p.add_subparsers(dest='cmd', required=True)

    def gpu(sp):
        sp.add_argument('--gpu-max-util', type=float, default=80.0,
                        help='GPU utilisation cap (%%), enforced by a pause after each batch (duty cycle)')
        sp.add_argument('--gpu-max-mem-gb', type=float, default=6.5, help='GPU memory cap (GB)')

    def common(sp):
        sp.add_argument('--layer', default='scan50-1950')
        sp.add_argument('--region')
        sp.add_argument('--bbox', help='lon0,lat0,lon1,lat1')
        sp.add_argument('--zoom', type=int)
        sp.add_argument('--workers', type=int, default=4)
        sp.add_argument('--dl-workers', type=int, default=4)
        sp.add_argument('--max-rps', type=float, default=8.0)

    s = sub.add_parser('dataset')
    common(s)
    s.add_argument('--name')
    s.add_argument('--eras', default='1940,1950')
    s.add_argument('--max-chips', type=int, default=0)
    s.add_argument('--neg-ratio', type=float, default=0.1)

    s = sub.add_parser('train')
    gpu(s)
    s.add_argument('--dataset', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--arch', default='unet')
    s.add_argument('--encoder', default='resnet34')
    s.add_argument('--init')
    s.add_argument('--epochs', type=int, default=30)
    s.add_argument('--batch', type=int, default=16)
    s.add_argument('--crop', type=int, default=384)
    s.add_argument('--lr', type=float, default=1e-3)
    s.add_argument('--pos-m', type=float, default=12.0)
    s.add_argument('--band-m', type=float, default=50.0)
    s.add_argument('--pos-weight', type=float, default=2.0)
    s.add_argument('--tol-m', type=float, default=10.0)
    s.add_argument('--eval-every', type=int, default=2)
    s.add_argument('--workers', type=int, default=4)
    s.add_argument('--shifts')

    s = sub.add_parser('offsets')
    gpu(s)
    s.add_argument('--dataset', required=True)
    s.add_argument('--model', required=True)
    s.add_argument('--batch', type=int, default=16)
    s.add_argument('--search-m', type=float, default=48.0)
    s.add_argument('--method', default='corr', choices=['corr', 'profile'])
    s.add_argument('--min-conf', type=float, default=0.4)

    s = sub.add_parser('extract')
    common(s)
    gpu(s)
    s.add_argument('--model', required=True)
    s.add_argument('--run', required=True)
    s.add_argument('--eras', default='1940,1950')
    s.add_argument('--lines', help='line ids (comma-separated)')
    s.add_argument('--targets', default='coarse', choices=['coarse', 'eval', 'file'])
    s.add_argument('--targets-file', help='--targets file: prepared targets (rail.py junctions), e.g. /work/junctions/targets-coarse.jsonl')
    s.add_argument('--no-predict', action='store_true', help='only use probabilities already computed (no GPU)')
    s.add_argument('--degrade-m', type=float, default=0, help='eval: simplify the coarse trace (Douglas-Peucker, m) to simulate chords')
    s.add_argument('--eval-scope', default='truth', choices=['truth', 'closed', 'all'], help='eval: lines in service at the time (truth), closed before (closed), or all')
    s.add_argument('--batch', type=int, default=16)
    s.add_argument('--limit', type=int, default=0)
    s.add_argument('--no-download', action='store_true')
    s.add_argument('--download-only', action='store_true')
    s.add_argument('--exclude-networks', help='networks excluded from the targets (e.g. militaire)')
    s.add_argument('--tta', type=int, default=4, choices=[1, 2, 4], help='rotations averaged at inference')

    s = sub.add_parser('evaluate')
    s.add_argument('--run', required=True)
    s.add_argument('--junctions', action='store_true', help='"jonction-eval" run: accuracy after the 15 m cut')

    s = sub.add_parser('junctions', help='junction places between coarse and precise traces, and corridors (read-only)')
    s.add_argument('--mode', default='coarse', choices=['coarse', 'eval'])
    s.add_argument('--name', help='file suffix (default: the mode)')
    s.add_argument('--detector', default='/reports/jonctions-detecteur/defauts_*.csv', help='national detector CSV (glob)')
    s.add_argument('--exclude-networks', default='militaire')
    s.add_argument('--sample', type=int, default=0, help='number of randomly drawn pieces (eval mode)')

    s = sub.add_parser('write')
    s.add_argument('--run', required=True, help='one or more runs, comma-separated')
    s.add_argument('--min-conf', type=float, default=0.7)
    s.add_argument('--eval-ref', help='evaluation-ref.json (estimated offset) if missing from the runs')
    s.add_argument('--replace-all', action='store_true', help='replaces ALL ign-historique pieces of the layer (national run)')
    s.add_argument('--no-vertex-check', action='store_true', help='without the "vertices" filter (not recommended: 18 %% false positives on closed lines)')
    s.add_argument('--dry-run', action='store_true')

    s = sub.add_parser('evalchips', help='pixel F1 on chip sets (check outside training)')
    gpu(s)
    s.add_argument('--model', required=True)
    s.add_argument('--dataset', required=True, help='comma-separated sets')
    s.add_argument('--tol-m', type=float, default=6.0)

    s = sub.add_parser('export', help='safetensors (fp32, fp16) + ONNX + SHA-256 for publication')
    s.add_argument('--model', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--check-dataset', help='control sets to compare fp16 and fp32')
    s.add_argument('--no-onnx', action='store_true', help='safetensors only (no onnx / onnxruntime)')
    s.add_argument('--tol-m', type=float, default=6.0)

    a = p.parse_args()
    if a.cmd == 'dataset':
        from vffrail import dataset
        a.bbox = bbox_arg(a)
        a.name = a.name or a.region or 'zone'
        dataset.build(a)
    elif a.cmd == 'train':
        from vffrail import train
        train.run(a)
    elif a.cmd == 'offsets':
        from vffrail import offsets
        offsets.run(a)
    elif a.cmd == 'extract':
        from vffrail import extract
        a.bbox = bbox_arg(a)
        extract.run(a)
    elif a.cmd == 'evaluate':
        if a.junctions:
            from vffrail import jwrite
            jwrite.evaluate_junctions(a)
        else:
            from vffrail import extract
            extract.evaluate(a)
    elif a.cmd == 'junctions':
        from vffrail import junctions
        a.name = a.name or a.mode
        junctions.build(a)
    elif a.cmd == 'evalchips':
        from vffrail import export
        export.evalchips(a)
    elif a.cmd == 'export':
        from vffrail import export
        export.export(a)
    elif a.cmd == 'write':
        from vffrail import extract
        extract.write(a)


if __name__ == '__main__':
    main()
