# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake ninja-build git curl ca-certificates ffmpeg libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade "pip>=24" "setuptools>=70.1" wheel packaging

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch torchvision torchaudio

RUN python3 -c "import torch; print('torch:', torch.__version__)"

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir psutil cython

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir --no-build-isolation flash-attn

COPY requirements*.txt /app/

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /app/requirements.txt \
                           -r /app/requirements_animate.txt \
                           -r /app/requirements_s2v.txt

CMD ["bash"]
