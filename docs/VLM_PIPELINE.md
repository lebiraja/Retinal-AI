# VLM Pipeline

This document describes how the Vision-Language Model (VLM) is used in production.

## Provider

The backend uses local Ollama HTTP API.

- Endpoint: `POST /api/chat`
- Model: `gemma4:e2b` by default
- Base URL in Docker: `http://ollama:11434`

Implementation is in `services/backend/src/services/vlm_service.py`.

## Two-Stage Design

The backend uses two distinct VLM stages.

## Stage 1: Eye Image Gate (Fail Closed)

Function: `check_is_eye_image(image_bytes, content_type)`

Behavior:

- Sends image to VLM with strict YES/NO prompt.
- Expects one-word answer.
- Returns `True` when answer starts with `YES`.
- Returns `False` when answer does not start with `YES`.
- Retries with exponential backoff on timeout/transport failures.
- Raises `VLMUnavailableError` if all retries fail.

API outcomes:

- VLM says `NO` -> backend returns `422`.
- VLM unavailable -> backend returns `503`.
- VLM says `YES` -> continue to CNN inference.

This prevents non-eye images from reaching the disease classifier.

## Stage 2: Personalized Advisory (Fail Open)

Function: `generate_analysis(...)`

Behavior:

- Uses detected diseases, probabilities, and risk level from CNN.
- Sends both image and summary context to VLM.
- Requests concise patient-friendly advisory text.
- If VLM call fails, backend falls back to static advisory text.

This means prediction still succeeds even if advisory generation fails.

## Image Optimization

Before VLM calls, uploaded images are preprocessed:

- Convert to RGB
- Downscale max side to `VLM_MAX_IMAGE_SIDE` (default `768`)
- Encode as JPEG (quality 85)
- Base64 encode for Ollama request payload

Goal: reduce VLM latency and payload size without losing key retinal context.

## Tuning Knobs

Gate path:

- `VLM_GATE_TIMEOUT_S`
- `VLM_GATE_MAX_RETRIES`
- `VLM_GATE_MAX_TOKENS`

Analysis path:

- `VLM_ANALYSIS_TIMEOUT_S`
- `VLM_ANALYSIS_MAX_TOKENS`
- `OLLAMA_TEMPERATURE`

Global:

- `OLLAMA_TIMEOUT_S`
- `VLM_MAX_IMAGE_SIDE`

## Reliability Model

- Gate is strict by default (`VLM_GATE_REQUIRED=true`).
- Advisory generation is best-effort and non-blocking for core prediction quality.
- Backend maintains graceful fallback for advisory text when stage 2 fails.

## Where to Trace in Logs

Backend logs include:

- Gate result with request ID and attempt count
- Model inference failures vs VLM failures
- End-to-end request time in milliseconds

Primary files:

- `services/backend/src/services/vlm_service.py`
- `services/backend/src/routers/predict.py`
- `services/backend/src/config.py`
