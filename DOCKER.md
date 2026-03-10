# Retinal Disease Classifier — Docker & Microservices Guide

## Architecture Overview

The system is split into **four independent Docker services** communicating over an isolated Docker bridge network:

```
Internet
    │
    ▼
┌─────────────┐   port 80
│    nginx    │ ◄─────────── public entry point
│  (proxy)   │
└──────┬──────┘
       │
       ├── /api/* ──────────►  ┌───────────┐
       │                       │  backend  │  port 8000 (internal)
       │                       │ (FastAPI) │
       │                       └─────┬─────┘
       │                             │  HTTP /inference
       │                             ▼
       │                       ┌─────────────────┐
       │                       │  model-service  │  port 8001 (internal)
       │                       │  (FastAPI+GPU)  │
       │                       └─────────────────┘
       │
       └── /* ──────────────►  ┌──────────────┐
                                │   frontend   │  port 80 (internal)
                                │  (React+Nginx)│
                                └──────────────┘
```

### Services

| Service | Technology | Port | GPU | Purpose |
|---------|-----------|------|-----|---------|
| `nginx` | Nginx 1.27 | **80** (public) | No | Reverse proxy, rate limiting, security headers |
| `backend` | FastAPI + Python 3.11 | 8000 (internal) | No | Validation, advisory logic, API gateway |
| `model-service` | FastAPI + PyTorch + CUDA 12.1 | 8001 (internal) | **Yes** | EfficientNet-B4 inference |
| `frontend` | React 19 + Vite + Nginx | 80 (internal) | No | React SPA static file server |

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Docker | ≥ 26 | `docker --version` |
| Docker Compose | ≥ 2.20 | `docker compose version` |
| NVIDIA Driver | ≥ 525 | `nvidia-smi` |
| nvidia-container-toolkit | latest | `nvidia-ctk --version` |

### Install NVIDIA Container Toolkit (Ubuntu/Debian)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

> **Verify GPU access in Docker:**
> ```bash
> docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
> ```

---

## Quick Start

```bash
# 1. Clone (or enter) the project root
cd /path/to/mindcraft-2k26

# 2. Copy and configure environment
cp .env.example .env
# Edit .env if you need to change HF_MODEL_REPO or CORS_ORIGINS

# 3. Build and start all services
make up-d

# 4. Check everything is healthy
make health
```

Then open **http://localhost** in your browser.

---

## Building Images

All images are built from the **project root** (not inside `services/`).
The Dockerfiles use `COPY services/...` and `COPY cnn/...` so the build
context must include the whole project.

```bash
# Build all images
docker compose build

# Build a single service
docker compose build model-service
docker compose build backend
docker compose build frontend
docker compose build nginx
```

## Starting Services

```bash
# Development (includes docker-compose.override.yml automatically)
docker compose up --build

# Production (no dev overrides, detached)
docker compose -f docker-compose.yml up --build -d

# With Makefile shortcuts
make up           # foreground (dev)
make up-d         # detached   (dev)
make up-prod      # detached   (prod)
```

## Stopping Services

```bash
docker compose down              # stop containers
docker compose down -v           # stop + remove volumes (deletes HF model cache!)
```

---

## Service Details

### model-service

- **Base image**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
- **GPU**: Required (will fail gracefully and fall back to CPU if no GPU is found)
- **Startup time**: 2–10 minutes on first run (downloads ~380 MB model from HuggingFace)
- **Subsequent starts**: ~10–30 seconds (model is cached in Docker volume `retinal_hf_cache`)

#### Skip HuggingFace download (air-gapped deployment)

If you already have `best_model.pt` locally:

```yaml
# In docker-compose.yml, under model-service:
environment:
  - MODEL_CHECKPOINT_PATH=/checkpoints/best_model.pt
volumes:
  - ./outputs/checkpoints:/checkpoints:ro
```

#### GPU metrics

```bash
# Check GPU usage inside the container
docker exec retinal-model-service nvidia-smi
# Or via Makefile
make gpu-check
```

### backend

- **Base image**: `python:3.11-slim`
- **No ML dependencies** — torch is NOT installed; only httpx + FastAPI + Pillow
- **Workers**: 2 Uvicorn workers (configurable via `BACKEND_WORKERS`)
- **Swagger UI**: http://localhost/docs

### frontend

- **Multi-stage build**: Node 20 builds the React app, then Nginx serves the static output
- **API routing**: `VITE_API_URL=/api` is baked into the bundle at build time
- All requests to `/api/*` are proxied through Nginx to the backend

### nginx (reverse proxy)

- Rate limits: 30 req/min for `/api/predict`, 10 req/min for `/api/predict-batch`
- Sets security headers (X-Frame-Options, CSP, etc.)
- Handles client-side routing fallback (React Router)

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_REPO` | `lebiraja/retinal-disease-classifier` | HuggingFace model repo ID |
| `HF_MODEL_FILE` | `pytorch_model.bin` | Checkpoint filename in the repo |
| `MODEL_CHECKPOINT_PATH` | *(unset)* | Local checkpoint path (overrides HF download) |
| `DEFAULT_THRESHOLD` | `0.5` | Default sigmoid threshold |
| `MAX_FILE_SIZE_MB` | `10` | Max upload size in MB |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `LOG_LEVEL` | `info` | `debug \| info \| warning \| error` |
| `HTTP_PORT` | `80` | Host port for Nginx |

---

## Monitoring & Debugging

### Logs

```bash
make logs          # all services
make logs-model    # GPU inference logs
make logs-backend  # API gateway logs
make logs-nginx    # access logs
```

### Health Checks

```bash
# nginx
curl http://localhost/nginx-health

# backend API (via nginx)
curl http://localhost/api/health

# model service (direct, only in dev with port 8001 exposed)
curl http://localhost:8001/health

# model service GPU metrics
curl http://localhost:8001/metrics
```

### API Documentation

- **Swagger UI**: http://localhost/docs
- **ReDoc**: http://localhost/redoc

### Running Tests

```bash
# From project root
PYTHONPATH=. pytest services/backend/tests/ -v
```

---

## Production Checklist

- [ ] Set `CORS_ORIGINS` to your frontend domain (not `*`)
- [ ] Add SSL/TLS termination (Certbot + Nginx, or a load balancer)
- [ ] Restrict `MODEL_SERVICE_URL` / `backend` to internal network only
- [ ] Set `LOG_LEVEL=warning` to reduce log volume
- [ ] Monitor GPU memory with `nvidia-smi dmon` or Prometheus/Grafana
- [ ] Enable Docker log rotation (already configured in `docker-compose.yml`)
- [ ] Back up the `retinal_hf_cache` Docker volume if going air-gapped

---

## Directory Structure (new microservices layout)

```
mindcraft-2k26/
├── services/
│   ├── model/                  GPU inference service
│   │   ├── src/
│   │   │   ├── config.py       Environment-driven configuration
│   │   │   ├── main.py         FastAPI app (POST /inference, GET /health)
│   │   │   ├── model.py        EfficientNet-B4 builder + checkpoint loader
│   │   │   ├── preprocess.py   Image preprocessing pipeline
│   │   │   └── schemas.py      Pydantic request/response models
│   │   ├── Dockerfile          nvidia/cuda base → Python 3.11 → PyTorch GPU
│   │   └── requirements.txt
│   │
│   ├── backend/                API gateway (no GPU)
│   │   ├── src/
│   │   │   ├── config.py
│   │   │   ├── main.py         FastAPI app with CORS + request ID middleware
│   │   │   ├── schemas.py
│   │   │   ├── routers/
│   │   │   │   └── predict.py  /predict, /predict-batch, /health, /info
│   │   │   └── services/
│   │   │       ├── advisory_service.py   Risk level + advisory text
│   │   │       ├── model_client.py       Async httpx client → model service
│   │   │       └── validation_service.py Upload + fundus validation
│   │   ├── tests/
│   │   │   └── test_backend.py
│   │   ├── Dockerfile          python:3.11-slim (no torch)
│   │   └── requirements.txt
│   │
│   └── frontend/               React SPA
│       ├── nginx.conf          Nginx config for static file serving
│       └── Dockerfile          Multi-stage: node:20 build + nginx:alpine serve
│
├── nginx/                      Reverse proxy
│   ├── nginx.conf              Proxy config, rate limiting, security headers
│   └── Dockerfile
│
├── cnn/retinal-frontend/       React source (Vite + React 19)
├── docker-compose.yml          Production compose
├── docker-compose.override.yml Dev overrides (exposed ports, source mounts)
├── .env.example
├── .dockerignore
└── Makefile                    Developer shortcuts
```
