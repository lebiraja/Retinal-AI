# Backend Guide

This document explains the backend gateway in `services/backend` and how it coordinates VLM and CNN inference.

## Responsibilities

The backend service does not run the CNN directly. It orchestrates:

- request parsing and validation
- eye-image gating via Ollama VLM
- inference calls to `model-service`
- risk summarization and advisory formatting
- fallback behavior and HTTP error mapping

## Main Source Files

- `services/backend/src/main.py`: app startup and router registration
- `services/backend/src/config.py`: runtime environment configuration
- `services/backend/src/routers/predict.py`: `/predict` and `/predict-batch` flow
- `services/backend/src/services/model_client.py`: HTTP client for model-service
- `services/backend/src/services/vlm_service.py`: Ollama gate and analysis
- `services/backend/src/services/advisory_service.py`: static advisory and risk logic
- `services/backend/src/services/validation_service.py`: upload size checks
- `services/backend/src/schemas.py`: API response models

## Request Pipeline

For each image, backend executes:

1. Read bytes and validate size (`MAX_FILE_SIZE_MB`).
2. Run VLM gate with strict yes/no response.
3. If gate passes, call model-service inference endpoint.
4. Build disease/risk summary from probabilities and threshold.
5. Attempt VLM personalized advisory generation.
6. Return final response with timing and threshold metadata.

## Fail-Close and Fail-Open Rules

Gate stage (`check_is_eye_image`): fail closed.

- explicit non-eye -> `422 Unprocessable Entity`
- VLM unavailable after retries -> `503 Service Unavailable`

Analysis stage (`generate_analysis`): fail open.

- if analysis fails, backend returns static advisory text
- classification response still succeeds if CNN inference succeeded

## Key Endpoints

- `POST /predict`
- `POST /predict-batch`
- `GET /health`
- `GET /info`

`/predict` accepts query param `threshold` in `[0.0, 1.0]`.

## Response Shape Highlights

Prediction response includes:

- `disease_risk`
- `predictions`
- `detected_diseases`
- `detected_diseases_full`
- `risk_level`
- `advisory`
- `elapsed_ms`
- `threshold`

## Environment Variables

Important backend runtime vars are defined in `services/backend/src/config.py`.

VLM:

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OLLAMA_TIMEOUT_S`
- `VLM_GATE_TIMEOUT_S`
- `VLM_ANALYSIS_TIMEOUT_S`
- `VLM_GATE_MAX_RETRIES`
- `VLM_GATE_MAX_TOKENS`
- `VLM_ANALYSIS_MAX_TOKENS`
- `VLM_GATE_REQUIRED`
- `VLM_MAX_IMAGE_SIDE`

Model-service client:

- `MODEL_SERVICE_URL`
- `MODEL_CONNECT_TIMEOUT`
- `MODEL_READ_TIMEOUT`
- `MODEL_MAX_RETRIES`

General:

- `DEFAULT_THRESHOLD`
- `MAX_FILE_SIZE_MB`
- `CORS_ORIGINS`
- `BACKEND_HOST`
- `BACKEND_PORT`

## Operational Notes

- Keep `VLM_GATE_REQUIRED=true` in production if non-eye filtering is required.
- If latency is high, tune token caps and image size before increasing timeouts.
- Use backend logs to separate gate failures, model failures, and advisory-only failures.
