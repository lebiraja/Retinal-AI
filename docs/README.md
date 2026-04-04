# Documentation Index

This folder contains separated documentation for training, inference, backend services, deployment, and operations.

## Start Here

- `docs/SETUP.md`: environment and prerequisites
- `docs/DEPLOYMENT.md`: Docker deployment modes and runtime checks
- `docs/USER_GUIDE.md`: user-facing app usage
- `docs/API_REFERENCE.md`: API endpoint and code reference

## Runtime and Architecture

- `docs/SYSTEM_ARCHITECTURE.md`: service topology and request flow
- `docs/BACKEND.md`: backend gateway behavior and endpoint pipeline
- `docs/VLM_PIPELINE.md`: Ollama gate plus advisory generation design
- `docs/VALIDATION_SERVICE.md`: upload size and gate validation behavior
- `docs/TROUBLESHOOTING.md`: operational failure modes and fixes

## ML and Model Docs

- `docs/ARCHITECTURE.md`: CNN model architecture and design choices
- `docs/TRAINING.md`: training workflow and hyperparameters
- `docs/INFERENCE.md`: inference details and output interpretation
- `docs/MODEL_CARD.md`: model metadata and limits
- `docs/DEVELOPER.md`: developer setup and contribution workflow

## Compose Files Covered

- `docker-compose.yml`: full stack, GPU-ready model-service
- `docker-compose.cpu.yml`: standalone CPU deployment stack
- `docker-compose.override.yml`: local override settings

## Current Runtime Defaults

- Public UI: `http://localhost:7000`
- Backend API (CPU compose direct): `http://localhost:7002`
- Ollama API: `http://localhost:11434`
- Default VLM model: `gemma4:e2b`

## Notes

- The backend now uses local Ollama instead of cloud VLM providers.
- Eye-image gating is mandatory by default and fails closed.
- Personalized advisory generation fails open and falls back to static advisory text.
