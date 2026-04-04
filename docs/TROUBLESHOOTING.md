# Troubleshooting

This guide focuses on current Docker + Ollama + backend runtime issues.

## Service Health Basics

Check stack status:

```bash
docker compose ps
```

Check CPU stack status:

```bash
docker compose -f docker-compose.cpu.yml ps
```

Tail logs:

```bash
docker compose logs -f backend ollama model-service nginx frontend
```

## Issue: Backend Returns 503 for Uploads

Symptom:

- API response says image verification service is temporarily unavailable.

Cause:

- VLM gate failed after retries.

Checks:

```bash
docker compose logs -f ollama
curl http://localhost:11434/api/tags
```

Fix:

- Ensure Ollama service is healthy.
- Confirm model exists (`gemma4:e2b` by default).
- If needed, rerun stack so `ollama-pull` completes successfully.

## Issue: Backend Returns 422 for Valid Image

Symptom:

- Upload rejected as non-eye image.

Cause:

- VLM gate judged image as non-ophthalmic.

Fix:

- Upload clear fundus/OCT/slit-lamp eye image.
- Avoid screenshots, compressed thumbnails, or heavily edited images.
- If needed, adjust gate prompt or model selection in `services/backend/src/services/vlm_service.py`.

## Issue: Ollama Model Pull Fails

Symptom:

- `ollama-pull` exits with network resolution errors.

Fix:

- Verify host internet and DNS.
- Keep DNS entries set in compose (`1.1.1.1`, `8.8.8.8`).
- Retry container start after network recovers.

## Issue: Frontend Not Reachable

Symptom:

- `http://localhost:7000` not loading.

Checks:

```bash
docker compose ps
curl -f http://localhost:7000/nginx-health
```

Fix:

- Ensure `frontend` and `nginx` are running and healthy.
- Verify port `7000` is not already used by another process.

## Issue: Slow Responses (30-120s)

Cause:

- CPU VLM inference latency.

Mitigations:

- Reduce `VLM_ANALYSIS_MAX_TOKENS`.
- Reduce `VLM_MAX_IMAGE_SIDE`.
- Keep advisory stage fail-open (default behavior).
- Use GPU where available for CNN path.

## Issue: Model-Service Unavailable (502)

Symptom:

- Backend returns model service unavailable.

Checks:

```bash
docker compose logs -f model-service
```

Fix:

- Wait for model download and startup warmup.
- Verify health endpoint inside container network.
- For GPU mode, verify NVIDIA container runtime is installed.

## Quick Reset

If services are in inconsistent state:

```bash
docker compose down
docker compose up --build -d
```

CPU mode reset:

```bash
docker compose -f docker-compose.cpu.yml down
docker compose -f docker-compose.cpu.yml up --build -d
```
