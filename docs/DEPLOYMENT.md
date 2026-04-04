# Deployment Guide

This guide describes how to run the full application in Docker, including the local Ollama VLM model.

## Services

The runtime stack uses these services:

- `ollama`: local vision-language model server (default model `gemma4:e2b`)
- `ollama-pull`: one-shot init container that pulls the configured model
- `model-service`: EfficientNet-B4 inference service
- `backend`: FastAPI API gateway and orchestration
- `frontend`: React UI served by Nginx
- `nginx`: public entry point and reverse proxy

## Ports

- `7000`: public web app via Nginx (`http://localhost:7000`)
- `7002`: backend API direct access in CPU compose (`http://localhost:7002`)
- `11434`: Ollama API (`http://localhost:11434`)

## Main Compose (GPU Preferred)

Use when NVIDIA runtime is available and you want fastest CNN inference.

```bash
docker compose up --build -d
```

Stop:

```bash
docker compose down
```

## CPU Compose (No GPU Required)

Use for machines without NVIDIA GPU runtime.

```bash
docker compose -f docker-compose.cpu.yml up --build -d
```

Stop:

```bash
docker compose -f docker-compose.cpu.yml down
```

## Environment Variables

Important runtime variables:

- `OLLAMA_BASE_URL` default: `http://ollama:11434`
- `OLLAMA_MODEL` default: `gemma4:e2b`
- `OLLAMA_TEMPERATURE` default: `0.2`
- `OLLAMA_TIMEOUT_S` default: `90.0`
- `VLM_GATE_REQUIRED` default: `true`
- `VLM_GATE_TIMEOUT_S` default: `35.0`
- `VLM_ANALYSIS_TIMEOUT_S` default: `35.0`
- `VLM_GATE_MAX_TOKENS` default: `16`
- `VLM_ANALYSIS_MAX_TOKENS` default: `90`
- `VLM_MAX_IMAGE_SIDE` default: `768`

Model service variables:

- `MODEL_SERVICE_URL` default: `http://model-service:8001`
- `MODEL_CONNECT_TIMEOUT` default: `5.0`
- `MODEL_READ_TIMEOUT` default: `90.0`
- `MODEL_MAX_RETRIES` default: `2`

## Health Checks

Useful checks after startup:

```bash
curl -f http://localhost:7000/nginx-health
curl -f http://localhost:7002/health
curl -f http://localhost:11434/api/tags
```

Container status:

```bash
docker compose ps
```

## Logs

Tail all logs:

```bash
docker compose logs -f
```

Tail backend only:

```bash
docker compose logs -f backend
```

Tail Ollama only:

```bash
docker compose logs -f ollama
```

## Common Startup Sequence

1. `ollama` starts and becomes healthy.
2. `ollama-pull` downloads `OLLAMA_MODEL`.
3. `model-service` becomes healthy.
4. `backend` starts after dependencies are ready.
5. `frontend` and `nginx` become healthy.

If `ollama-pull` fails, backend will not start because the model is not guaranteed to exist.
