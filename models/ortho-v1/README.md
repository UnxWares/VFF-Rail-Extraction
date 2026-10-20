# Model card: ortho-v1

Railway track centreline segmentation on historical IGN aerial photographs
(France, 1950-1965). Trained for [VFF](https://vff-project.org).

## Files

| file | size | SHA-256 |
|---|---|---|
| `ortho-v1.fp16.safetensors` | 49,173,856 bytes (49.2 MB) | `eb70890c0ff99ce5886c677baed338f11ca597ff5ac5804c978cb238a0db01f4` |
| `history.json` | training history (loss and validation metrics per epoch) | |

The weights were converted from the training checkpoint to safetensors (no pickle,
no executable code at load time) and cast to fp16. The safetensors header carries
the metadata (`arch`, `encoder`, `input`, `mean`, `std`, `layer`, `zoom`, `metrics`).

## Architecture

- U-Net, encoder ResNet-34 (initialised from ImageNet weights, torchvision),
  decoder attention scSE, 1 output channel (logit).
- Library: `segmentation-models-pytorch` 0.5:
  `smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=1, decoder_attention_type='scse')`.
- 338 tensors, about 24.6 M values (parameters and batch-norm statistics).

## Input and output

- Input: RGB, values 0-255, normalised with ImageNet statistics
  (mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`), shape
  (N, 3, H, W) with H and W multiples of 32. Trained on 384 px crops of 512 px chips.
- Imagery: IGN WMTS layer `ORTHOIMAGERY.ORTHOPHOTOS.1950-1965`
  (style `BDORTHOHISTORIQUE`), **zoom 16 Web Mercator, about 1.6 m per pixel** in France.
- Output: logit; after a sigmoid, the probability that the pixel lies within
  +/- 3 m of a track centreline. Optional test-time augmentation: average over
  4 rotations.

## Training

- 1,653 auto-labelled chips (512 x 512 px, about 7,000 z16 tiles) in Brittany, for
  lines in service in both 1950 and 1960.
- Labels: precise track axes from OpenStreetMap, IGN BD TOPO and validated manual
  corrections (and coarse traces already matched to one of them), rasterised as a
  +/- 3 m positive band, a +/- 7 m ignored band, other possible tracks ignored.
- Loss: masked BCE + Dice; AMP, OneCycle, 24 epochs, rotations, flips, hue jitter;
  about 25 min on an RTX 4060 capped at 80 %.
- Validation: 16 x 16 tile spatial blocks (1 in 5), never seen in training.

## Measured accuracy

Validation blocks (Brittany), relaxed metrics at 6 m:

| metric | value |
|---|---|
| F1 | 0.836 |
| precision / recall | 0.83 / 0.84 |
| median offset of the predicted skeleton | 2.2 m |
| p90 offset | 6.5 m |

End to end (extraction in a corridor around a coarse trace, lines held out):

- extracted pieces: median 2.2 m, p90 9.1 m; precision 0.92 at 10 m, 0.98 at 20 m;
  recall 0.82 at 10 m;
- coarse traces simplified to 40 m chords: coarse 8.7 m (p90 29.4 m) -> extracted
  4.6 m (p90 13.2 m);
- **at junctions**: median 2.7 m, p90 12.7 m.

## Limits

- Measured in Brittany only, applied elsewhere without regional measurements.
- Lines closed before 1950: recall 16 %; without filters 18 % of extracted km are
  more than 20 m off (roads, hedges, canals followed instead of the track); the
  vertex and chord filters bring this to about 2 %.
- The ortho photographs are not re-georeferenced; double track, stations, parallel
  roads and canals are the main sources of error. The probability is a plateau
  (14 to 33 m wide on multiple tracks), so the axis is taken at the plateau centre.
- Floor measured on the ground truth itself: median 2.7 m, p90 8.5 m. A 1950-1965
  photograph alone cannot guarantee p90 <= 5 m everywhere.
- Not suitable for scanned maps (the 1950 SCAN 50 drawing is offset 21 m median).
- Intended use: candidates for correcting coarse railway traces, reviewed or
  cross-checked with other sources. Not an authoritative survey.

## License and attributions

MIT © UnxWares. Credit when reusing the weights or their outputs:
IGN - BD ORTHO historique 1950-1965 (Licence Ouverte 2.0); labels derived from
OpenStreetMap (© OpenStreetMap contributors, ODbL 1.0) and IGN BD TOPO
(Licence Ouverte 2.0); ImageNet ResNet-34 initialisation from torchvision (BSD 3-Clause).
