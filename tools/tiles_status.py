"""Tile and probability status for a targets file (no GPU, no network).

    run.sh python -I /app/tools/tiles_status.py /work/junctions/targets-coarse.jsonl ortho-1950 ortho-v1 [output.jsonl]
With a 4th argument: writes the targets whose tiles all have their probability already (tests without GPU).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/app')
import numpy as np  # noqa: E402

from vffrail.extract import prob_path, prob_root_for, tiles_for_targets  # noqa: E402
from vffrail.tiles import LAYERS, TileCache  # noqa: E402

path, layer_key, model = sys.argv[1], sys.argv[2], sys.argv[3]
layer = LAYERS[layer_key]
targets = []
for row in open(path):
    t = json.loads(row)
    t['coarse'] = np.asarray(t['coarse'])
    targets.append(t)
tiles = tiles_for_targets(targets, layer.zoom)
cache = TileCache('/data', layer)
root = prob_root_for(Path('/work'), model, layer_key)
img = sum(1 for t in tiles if cache.path(layer.zoom, *t).exists())
none = sum(1 for t in tiles if cache.none_path(layer.zoom, *t).exists())
prob = sum(1 for t in tiles if prob_path(root, layer.zoom, *t).exists())
full = []
for t in targets:
    ts = tiles_for_targets([t], layer.zoom)
    if all(prob_path(root, layer.zoom, *x).exists() or cache.none_path(layer.zoom, *x).exists() for x in ts):
        full.append(t)
if len(sys.argv) > 4:
    with open(sys.argv[4], 'w') as fh:
        for t in full:
            t.pop('_chunks', None)
            fh.write(json.dumps({**t, 'coarse': t['coarse'].tolist()}) + '\n')
print(json.dumps({'cibles': len(targets), 'tuiles': len(tiles), 'images_en_cache': img, 'hors_couverture': none,
                  'a_telecharger': len(tiles) - img - none, 'probabilites': prob, 'a_predire': img - prob,
                  'cibles_entierement_predites': len(full)}))
