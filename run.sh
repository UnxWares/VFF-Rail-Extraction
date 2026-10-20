#!/usr/bin/env bash
# Runs the extraction tool in the container (Podman rootless, NVIDIA GPU through CDI).
#   ./run.sh build                       # build the image
#   ./run.sh gpu-check                   # torch.cuda.is_available()
#   ./run.sh extract --layer ortho-1950 --model ortho-v1 --region bretagne --targets eval --run eval-bretagne
#   ./run.sh python -I /app/tools/infer_tile.py ...
# Host network (PostgreSQL on 127.0.0.1:5432). CPU capped (--cpus) to keep the machine usable.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VFF_RAIL_IMAGE:-localhost/vff-rail:1}"
CPUS="${VFF_RAIL_CPUS:-12}"
TILES="${VFF_RAIL_TILES:-$HERE/data/tiles}"   # downloaded tiles (untrusted data)
WORK="${VFF_RAIL_WORK_DIR:-$HERE/data/work}"  # chips, models, probabilities, outputs
mkdir -p "$TILES" "$WORK/cache"

NET=(--network host --userns=keep-id)
[ -n "${VFF_RAIL_DBNET:-}" ] && NET=(--network "$VFF_RAIL_DBNET" --userns=keep-id -e DB_HOST=db)

if [ "${1:-}" = "build" ]; then
    exec podman build -t "$IMAGE" -f "$HERE/Containerfile" "$HERE"
fi

GPU=(--device nvidia.com/gpu=all)
[ "${VFF_RAIL_NO_GPU:-0}" = "1" ] && GPU=()

TTY=(-i)
[ -t 0 ] && [ -t 1 ] && TTY+=(-t)

# Optional environment file (database credentials, VFF_RAIL_COARSE_SOURCES...), never published.
ENVF=()
[ -f "$HERE/rail.local.env" ] && ENVF=(--env-file "$HERE/rail.local.env")

if [ "${1:-}" = "gpu-check" ]; then
    set -- python -I -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
elif [ "${1:-}" != "python" ] && [ "${1:-}" != "bash" ]; then
    set -- python -I /app/rail.py "$@"
fi

exec podman run --rm "${TTY[@]}" "${GPU[@]}" --security-opt label=disable "${NET[@]}" "${ENVF[@]}" \
    --cpus "$CPUS" --shm-size 4g \
    -e HOME=/work/cache/home -e TORCH_HOME=/work/cache/torch -e HF_HOME=/work/cache/hf \
    -e XDG_CACHE_HOME=/work/cache \
    -v "$HERE":/app:ro -v "$TILES":/data -v "$WORK":/work -w /work \
    "$IMAGE" "$@"
