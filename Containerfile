# Railway track extraction image (old IGN maps and photographs) - PyTorch CUDA.
#   podman build -t localhost/vff-rail:1 -f Containerfile .
#   podman run --rm --device nvidia.com/gpu=all --security-opt label=disable localhost/vff-rail:1 \
#       python -c "import torch; print(torch.cuda.is_available())"
# GPU: CDI (nvidia-container-toolkit, /etc/cdi/nvidia.yaml), no driver in the image.
FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Code mounted read-only on /app, data on /data (tiles) and /work (chips, models, outputs).
WORKDIR /work
