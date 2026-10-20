#!/usr/bin/env bash
# National extraction in DISJOINT cells (a piece belongs to the cell of its centroid):
# the next cell is downloaded (politely) while the previous one runs on the GPU.
#   ./france.sh ortho-1950 ortho-v1 1950,1960          # downloads + extracts
#   ./run.sh write --run $(ls -d …/runs/fr-ortho-1950-* | xargs -n1 basename | paste -sd,) \
#        --min-conf 0.7 --replace-all --eval-ref /work/runs/eval-ortho-bretagne-deg40/evaluation-ref.json
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAYER="${1:-ortho-1950}"
MODEL="${2:-ortho-v1}"
ERAS="${3:-1950,1960}"
TTA="${VFF_RAIL_TTA:-2}"
EXCL="${VFF_RAIL_EXCLUDE:-militaire}"   # 1914-1918 military tracks: barely visible in 1950-1965
WORK="${VFF_RAIL_WORK_DIR:-$HERE/data/work}"
LOGS="$WORK/logs"
mkdir -p "$LOGS"

LONS=(-5.3 -1.0 2.0 5.0 9.7)
LATS=(41.3 45.0 47.3 49.0 51.2)

prev=""
for ((i = 0; i < ${#LONS[@]} - 1; i++)); do
  for ((j = 0; j < ${#LATS[@]} - 1; j++)); do
    cell="${LONS[i]},${LATS[j]},${LONS[i+1]},${LATS[j+1]}"
    run="fr-${LAYER}-${i}${j}"
    if [ -f "$WORK/runs/$run/summary.json" ]; then
      echo "$(date +%T) cell $run already done"; continue
    fi
    echo "$(date +%T) cell $run ($cell): download"
    VFF_RAIL_NO_GPU=1 VFF_RAIL_CPUS=2 "$HERE/run.sh" extract --layer "$LAYER" --model none --bbox="$cell" --eras "$ERAS" \
      --targets coarse --exclude-networks "$EXCL" --run "dl-$run" --download-only --workers 2 > "$LOGS/dl-$run.log" 2>&1
    [ -n "$prev" ] && wait "$prev"
    echo "$(date +%T) cell $run: extraction"
    VFF_RAIL_CPUS=4 "$HERE/run.sh" extract --layer "$LAYER" --model "$MODEL" --bbox="$cell" --eras "$ERAS" \
      --targets coarse --exclude-networks "$EXCL" --run "$run" --workers 4 --batch 6 --tta "$TTA" --no-download > "$LOGS/$run.log" 2>&1 &
    prev=$!
  done
done
[ -n "$prev" ] && wait "$prev"
echo "$(date +%T) done"
