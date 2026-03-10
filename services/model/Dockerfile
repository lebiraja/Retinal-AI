# ─────────────────────────────────────────────────────────────────────────────
#  Model Service — GPU inference container
#
#  Base: python:3.11-slim (Debian) + PyTorch CUDA 12.1 wheels
#  The CUDA runtime libraries are pulled in automatically by the PyTorch wheel.
#  This avoids needing a heavy nvidia/cuda base image and sidesteps PPA issues.
#
#  Build:
#    docker build -t retinal/model-service:latest -f services/model/Dockerfile .
#
#  Run (with GPU):
#    docker run --gpus all -p 8001:8001 retinal/model-service:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ────────────────────────────────────────────────
FROM python:3.11-slim AS deps

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch GPU wheels (cu121) — large layer, cache-friendly if kept separate
# --timeout / --retries guard against slow mirrors
RUN pip install --no-cache-dir --timeout=600 --retries=5 \
    torch==2.3.1 \
    torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining application dependencies
COPY services/model/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: production image ──────────────────────────────────────────────────
FROM python:3.11-slim AS production

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system libraries required by OpenCV / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copy the entire Python site-packages (torch + deps) from builder
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY services/model/src ./services/model/src
# Ensure the package hierarchy is importable
RUN touch ./services/__init__.py ./services/model/__init__.py

# Create cache directory and hand ownership to appuser
RUN mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appgroup /app

# ── Environment variables (overridable at runtime) ────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface
ENV MODEL_SERVICE_HOST=0.0.0.0
ENV MODEL_SERVICE_PORT=8001
ENV LOG_LEVEL=info
# PYTORCH_CUDA_ALLOC_CONF reduces GPU memory fragmentation
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8001

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "-m", "uvicorn", \
     "services.model.src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--log-level", "info", \
     "--workers", "1"]
