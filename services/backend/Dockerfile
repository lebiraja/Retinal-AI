# ─────────────────────────────────────────────────────────────────────────────
#  Backend Service — API gateway (no GPU required)
#
#  Base   : python:3.11-slim (Debian Bookworm)
#  Python : 3.11
#  No ML dependencies — torch is intentionally NOT installed here.
#
#  Build:
#    docker build -t retinal/backend-service:latest ./services/backend
#
#  Run:
#    docker run -p 8000:8000 \
#      -e MODEL_SERVICE_URL=http://model-service:8001 \
#      retinal/backend-service:latest
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies into a virtual environment for easy copying
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY services/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: production image ──────────────────────────────────────────────────
FROM python:3.11-slim AS production

# Security: create non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy application source
COPY services/backend/src ./services/backend/src
# Make the package hierarchy importable
RUN touch ./services/__init__.py ./services/backend/__init__.py

RUN chown -R appuser:appgroup /app

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000
ENV LOG_LEVEL=info
ENV MODEL_SERVICE_URL=http://model-service:8001

EXPOSE 8000

USER appuser

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", \
     "services.backend.src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info", \
     "--workers", "2", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
