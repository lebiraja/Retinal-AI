# Validation and Rejection Rules

Validation is split between lightweight upload checks and VLM semantic checks.

## Where Validation Happens

- File: `services/backend/src/routers/predict.py`
- Size check: `services/backend/src/services/validation_service.py`
- Eye-image semantic gate: `services/backend/src/services/vlm_service.py`

## Stage 1: Upload Size Validation

Before any model calls, backend validates file size.

- Config key: `MAX_FILE_SIZE_MB`
- Default: `20`
- Behavior on failure: `413 Payload Too Large`

This prevents expensive processing for oversized uploads.

## Stage 2: VLM Eye-Image Gate

The gate asks Ollama whether the upload is an ophthalmic image.

Accepted examples include:

- retinal fundus photos
- OCT images
- slit-lamp eye images
- optic disc images

Rejected examples include:

- selfies
- screenshots
- documents
- memes and unrelated photos

### Gate outcomes

- `YES`: continue to CNN inference
- `NO`: return `422 Unprocessable Entity`
- timeout/transport failures after retries: return `503 Service Unavailable`

Config keys:

- `VLM_GATE_REQUIRED`
- `VLM_GATE_TIMEOUT_S`
- `VLM_GATE_MAX_RETRIES`
- `VLM_GATE_MAX_TOKENS`

## Why This Design

- Prevents non-eye inputs from generating meaningless disease predictions.
- Keeps invalid requests cheap by rejecting early.
- Keeps behavior explicit and debuggable through response codes.

## Advisory Stage Is Separate

After CNN inference, personalized advisory generation also uses VLM, but that stage is fail-open.

- If advisory generation fails, prediction still returns using static advisory text.
- This preserves availability for core classification results.
