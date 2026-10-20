# Making the track detection model more reliable (separate project)

**Status: not started.** Roadmap for a more reliable model (ortho-v2). ortho-v1 is
published as is, with its measured limits (Brittany only); this project starts if one of
the signals below appears (section 2).

## 1. Where the model stands

- **Model `ortho-v1`**: U-Net (Segmentation Models PyTorch), ResNet-34 encoder
  pre-trained on ImageNet, specialised by VFF on the IGN BD ORTHO historique 1950-1965
  aerial photographs.
- **Training data**: 1,653 chips of 512 px, all in **Brittany**, labelled automatically
  from lines whose exact trace is known.
- **Evaluation (Brittany only)**: F1 0.836; median offset 2.2 m, p90 9.1 m; simulated
  chords: coarse trace 8.7 m -> extracted 4.6 m; false detections on lines closed before
  1950: 18 % without a filter, 1.2 to 1.9 % with the "vertices + chord envelope" filter.
- **Discarded basemap**: the 1950 SCAN 50 (drawing offset by 21 m median, extraction at 27 m).
- **Use**: national extraction in progress over about 12,200 km of coarse traces, results
  stored as the `ign-historique` source.

**Main weakness**: the model has never been measured outside Brittany, yet it is applied
to the whole of France (mountains, cereal plains, Mediterranean, industrial basins,
urban areas).

## 2. When to start this project

Start it if one of these signals appears:

- a measurement outside Brittany gives a median offset above 3 m, a p90 above 12 m or
  more than 3 % false detections in a region;
- visual checks on the map show `ign-historique` traces that are clearly wrong (a road,
  canal or hedge followed instead of the track);
- `ign-historique` traces often contradict the other sources (OHM, other coarse traces,
  old maps) during merging;
- a more reliable version is to be published (section 6).

## 3. Measure before fixing

1. **Control lines per region**: 4 to 5 regions with different landscapes (mountains,
   cereal plain, hedged farmland, Mediterranean, industrial North), lines with a known
   trace, excluded from training and split into spatial blocks.
2. **Metrics per region and per case**: F1, median and p90 offset, false detections after
   filtering, missed detections; broken down by single / double track, tunnel / cutting /
   embankment, urban area, closure date.
3. **Human verification sample**: about 200 randomly drawn sections, checked by eye on
   the orthophoto, to estimate a real error rate with its confidence interval.

## 4. Reliability measures (in order of effort / gain)

1. **Multi-region training set (`ortho-v2`)**: sampling stratified by region and
   landscape, validation by spatial blocks, strict comparison with ortho-v1 on the same
   control lines.
2. **Hard negatives**: straight roads, canals, hedges, power lines, greenways, plot
   boundaries; added explicitly to the set.
3. **Label quality**:
   - correct the georeferencing offset of the 1950 orthophotos before labelling (the
     `offsets` command can measure it);
   - discard lines whose reference trace is doubtful;
   - handle double track (single axis or two axes).
4. **Augmentations suited to old photographs**: contrast, grain, blur, haze, black and
   white or sepia, scratches.
5. **Model**: larger encoders (ResNet-50, EfficientNet), UNet++, an ensemble of 2 to 3
   models; probability calibration to justify the `--min-conf 0.7` threshold.
6. **Post-processing**: topological continuity, minimum railway curvature radius
   (already 200 m for the filter), connection to neighbouring precise traces, rejection
   of small isolated pieces.
7. **Confidence per section**: store a score in `source_traces`, use it as a threshold
   when merging traces, and be able to display it.
8. **Other basemaps**: areas not covered by the 1950-1965 orthophotos; the Etat-Major
   map (1820-1866) for lines lifted before 1950, with a separate model; SCAN 50 only
   after its offset is corrected.

## 5. Code

- `pytest` tests of the pure functions (`geo`, `field`, `offsets`, `throttle`, cell
  splitting), runnable without a GPU, wired into CI.
- Reproducibility: fixed seed, pinned versions, dataset fingerprint recorded with each
  model.

## 6. Publishing a more reliable version

Conditions for publishing a more reliable model (ortho-v2):

- **Criteria**: in every tested region, median offset <= 3 m, p90 <= 10 m, false
  detections <= 2 %; known limits documented.
- **Model card**: intended use, data, architecture, metrics per region, limits,
  reproducible training command.
- **Exports**: safetensors (fp16 if accuracy is unchanged), ONNX, SHA-256 sums, example
  inference script. ortho-v1: fp16 weights (49.2 MB) in the repository, fp32 weights
  (98.3 MB) published elsewhere (link in the README).
- **Weights license (decided on 2026-10-20)**: MIT, with attribution of the IGN
  (Licence Ouverte 2.0, photos and BD TOPO) and of the OpenStreetMap contributors
  (labels partly derived from OSM, ODbL, as a produced work), and a mention of
  torchvision (BSD-3) for the initial weights.
- ortho-v1 (fp16, safetensors) published on 2026-10-20.

## 7. Resources

GPU capped at 80 % (`--gpu-max-util 80`), overall CPU capped at 80 %. Order of
magnitude: a few GPU hours for an ortho-v2 and its evaluations, plus downloading the
tiles of the control regions (polite, rate-limited by the IGN service).
