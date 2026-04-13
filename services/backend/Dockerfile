# ─────────────────────────────────────────────────────────────────────────────
#  Backend Service — API gateway (no GPU required)
#
#  Framework : Django 4.2 + Django REST Framework  (migrated from FastAPI)
#  Server    : Gunicorn + UvicornWorker (ASGI — supports async Django views)
#  Base      : python:3.11-slim (Debian Bookworm)
#  Python    : 3.11
#
#  No ML dependencies — torch is intentionally NOT installed here.
#
#  Build (from repo root):
#    docker build -f services/backend/Dockerfile -t retinal/backend:latest .
#
#  Run standalone (dev):
#    docker run -p 8000:8000 \
#      -e MODEL_SERVICE_URL=http://model-service:8001 \
#      retinal/backend:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install into a venv for clean copying to the production stage
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY services/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: production image ─────────────────────────────────────────────────
FROM python:3.11-slim AS production

# curl for HEALTHCHECK; non-root user for security
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 appgroup \
    && useradd  --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

WORKDIR /app

# ── Copy venv + source ────────────────────────────────────────────────────────
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Application source (keep same layout — PYTHONPATH=/app resolves everything)
COPY services/backend/src ./services/backend/src

# Ensure the package hierarchy is importable as 'services.backend.src.*'
RUN touch ./services/__init__.py ./services/backend/__init__.py

RUN chown -R appuser:appgroup /app

# ── Environment ───────────────────────────────────────────────────────────────
# PYTHONPATH=/app  →  'import services.backend.src.config' resolves correctly
# PYTHONPATH also includes the Django project root so 'import retinal_backend'
# resolves from /app/services/backend/src/retinal_backend/
ENV PYTHONPATH=/app/services/backend/src:/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DJANGO_SETTINGS_MODULE=retinal_backend.settings

# Service defaults (can be overridden in docker-compose or at runtime)
ENV LOG_LEVEL=info
ENV MODEL_SERVICE_URL=http://model-service:8001
ENV OLLAMA_BASE_URL=http://ollama:11434

EXPOSE 8000

USER appuser

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Gunicorn + UvicornWorker (ASGI) ──────────────────────────────────────────
# Replaces: python -m uvicorn services.backend.src.main:app --workers 2
# Django async views require ASGI — UvicornWorker provides the async event loop.
# 2 workers = same as the old FastAPI config; no GPU so multi-worker is safe.
CMD ["gunicorn", \
     "retinal_backend.asgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--forwarded-allow-ips", "*", \
     "--access-logfile", "-"]
