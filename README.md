# VFF Rail Extraction

Extract the centreline of railway tracks from historical IGN aerial photographs
(France, 1950-1965) with a segmentation network, then turn the probabilities into
precise track geometries, including at junctions.

Part of [VFF](https://vff-project.org), an open encyclopedia of the French railway
network ([main repository](https://github.com/UnxWares/VFF-Project)). VFF uses this
tool to correct the geometry of lines whose only known trace is coarse (hand-drawn,
generalised or simplified), especially closed and dismantled lines.

## What is in this repository

| path | content |
|---|---|
| `models/ortho-v1/ortho-v1.fp16.safetensors` | trained weights (fp16, 49.2 MB), see the [model card](models/ortho-v1/README.md) |
| `models/ortho-v1/SHA256SUMS` | checksum of the published weights |
| `vffrail/` | dataset building, training, inference, vectorisation, junction post-processing |
| `tools/infer_tile.py` | standalone inference on one image (no database needed) |
| `tests/` | unit tests (pure functions, no GPU) |
| `rail.py`, `run.sh`, `Containerfile` | command-line entry point and container |
| `RELIABILITY.md` | reliability roadmap |

## The model in short

- **Architecture**: U-Net with a ResNet-34 encoder (ImageNet initialisation) and scSE
  attention in the decoder ([segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)), 1 output class.
- **Input**: RGB 0-255, ImageNet normalisation (mean `[123.675, 116.28, 103.53]`,
  std `[58.395, 57.12, 57.375]`), height and width multiples of 32.
- **Imagery**: IGN WMTS layer `ORTHOIMAGERY.ORTHOPHOTOS.1950-1965`, zoom 16 Web Mercator
  (about 1.6 m per pixel in France).
- **Output**: per-pixel probability (after a sigmoid) that the pixel lies within
  +/- 3 m of a track centreline.
- **Measured accuracy** (validation spatial blocks never seen in training, Brittany):
  F1 0.836 (precision 0.83, recall 0.84 at 6 m), median offset 2.2 m, p90 6.5 m.
  At junctions: median 2.7 m, p90 12.7 m.

Details, end-to-end figures and limits: [model card](models/ortho-v1/README.md).

## Full-precision weights (fp32)

The fp32 version of ortho-v1 is not stored in this repository (link coming soon).

- size: 98.3 MB (98,312,928 bytes), file `ortho-v1.fp32.safetensors`;
- SHA-256: `24aed7a8d99c40b948c9a308ab6861deb1de5b41658b151c016fd15e219722f0`.

## Quick start

### 1. Container

Everything runs in a container (Podman rootless; the GPU is optional and goes
through CDI, `nvidia-container-toolkit`).

```bash
./run.sh build                     # image localhost/vff-rail:1 (PyTorch 2.8, CUDA 12.8)
./run.sh gpu-check                 # optional
(cd models/ortho-v1 && sha256sum -c SHA256SUMS)   # check the weights
```

`run.sh` mounts this directory read-only on `/app`, the tile cache on `/data`
(`VFF_RAIL_TILES`, default `./data/tiles`) and a working directory on `/work`
(`VFF_RAIL_WORK_DIR`, default `./data/work`). `VFF_RAIL_NO_GPU=1` runs on CPU,
`VFF_RAIL_CPUS` caps the CPU.

### 2. Inference on an image

```bash
VFF_RAIL_NO_GPU=1 ./run.sh python -I /app/tools/infer_tile.py \
    /app/models/ortho-v1/ortho-v1.fp16.safetensors /work/photo.png /work/prob.png --tta
```

The weights are stored in fp16; load them into an fp32 model (as `infer_tile.py`
does) or keep fp16 on GPU. Use imagery at the training scale (z16, about 1.6 m/px):
other scales or other photo campaigns have not been measured. Safetensors is used
instead of a pickled checkpoint, so loading the weights cannot execute code.

### 3. Full pipeline (needs a PostGIS database with the VFF schema)

The dataset, extraction and write steps read the VFF database (tables `lines`,
`line_segments`, `eras`, `line_reference_parts`, `source_traces`). Connection:
`DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD`, which can be
put in an unpublished `rail.local.env` file read by `run.sh`.

```bash
R=./run.sh
$R dataset --layer ortho-1950 --region bretagne --eras 1950,1960 --max-chips 1600      # auto-labelled chips
$R train   --dataset ortho-1950-bretagne --out ortho-v1 --pos-m 6 --band-m 14 --tol-m 6 --epochs 24
$R extract --layer ortho-1950 --model ortho-v1 --region bretagne --eras 1950,1960 --targets eval --run eval-bretagne
$R evaluate --run eval-bretagne
./france.sh ortho-1950 ortho-v1 1950,1960                                             # national run, disjoint cells
$R junctions --mode coarse ...                                                         # junction corridors
```

Tiles are downloaded politely from the IGN Geoplateforme (User-Agent
`VFF-Bot/1.0 (+https://vff-project.org)`, at most 8 requests per second,
back-off on 429 / 5xx, disk cache, only the tiles that are needed).

#### Coarse source names

The traces to correct are rows of `line_reference_parts` whose `source` is a
"coarse" source. Source names depend on the database, so they are configurable,
with neutral defaults:

```bash
export VFF_RAIL_COARSE_SOURCES="imprecise:120,ohm:120,sncf:200"   # name:corridor half-width (m)
```

or a `rail.local.json` file at the repository root (or `VFF_RAIL_CONFIG=/path.json`):

```json
{"coarse_sources": {"imprecise": 120, "ohm": 120, "sncf": 200}, "work_dir": "/work"}
```

### 4. Post-processing

- **Corridor and path**: around each coarse trace, a corridor (base half-width per
  source plus 15 % of the longest local chord, at most 400 m); minimum-cost path on
  `-log p` by overlapping 4 km chunks; only supported parts are kept (smoothed
  probability >= 0.35 over >= 200 m, curvature radius >= 60 m). Nothing is invented:
  elsewhere the coarse trace stays the reference.
- **False-positive filters**: vertex check (a kept piece passes within 15 m of every
  coarse vertex it covers) and chord envelope check (at most 5 % of points outside
  15 m plus the sagitta of a 200 m radius curve).
- **Junctions** (`vffrail/snap.py`): on photographs a junction is only certain to
  about 15 m. An extracted branch is cut at its first approach within **15 m**
  (`TOL_M`) of the precise partner track; the cut end carries a connection point
  (projection on the partner track) and both headings, so that a tangent connection
  can be drawn later.
- **Sub-pixel axis** (`vffrail/precise.py`): transverse profiles of the probability
  along the extracted path give the centre of the half-height plateau (the model is
  trained on a +/- 3 m band, so the response is a plateau, not a peak); a penalised
  robust fit (continuous curvature, minimum radius) gives the railway geometry; at
  switches, a curve tangent to the partner track is fitted; every station gets an
  estimated error and a strict filter keeps only reliable parts.

### 5. Tests

```bash
python -I -m unittest discover -s tests     # numpy, scipy, shapely, scikit-image, pillow
```

Tests that import `vffrail.extract` need PyTorch (run them in the container).

## Limits

- Trained and measured on **Brittany only**; it is applied to the whole of France
  without regional measurements yet (mountains, cereal plains, Mediterranean,
  industrial basins, cities). See `RELIABILITY.md`.
- Lines closed **before 1950**: low recall (16 %), because many trackbeds are
  invisible or reused by roads; without filters 18 % of extracted km were more than
  20 m off (roads, hedges). The filters reduce this to about 2 %.
- A single 1950-1965 photograph cannot guarantee p90 <= 5 m everywhere: measured on
  the ground truth itself, the plateau centre has a floor of median 2.7 m, p90 8.5 m.
- Georeferencing of the old photographs is not corrected; double track, stations and
  parallel roads or canals remain the main sources of error.
- Not usable on the 1950 SCAN 50 maps (the drawing is offset by 21 m median from the
  real axis).
- Outputs are **candidates for correcting** a coarse trace, not an authoritative
  source.

## License and attributions

Code and weights: **MIT** © UnxWares, see [LICENSE](LICENSE). The weights were
trained on open data; please credit, when you reuse them or their outputs:

- aerial photographs: **IGN - BD ORTHO historique 1950-1965**, Licence Ouverte /
  Open Licence 2.0;
- training labels derived from **OpenStreetMap** (© OpenStreetMap contributors,
  ODbL 1.0) and from the **IGN BD TOPO** (Licence Ouverte / Open Licence 2.0);
- encoder initialisation: ImageNet ResNet-34 from torchvision (BSD 3-Clause).
