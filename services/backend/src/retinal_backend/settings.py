"""
Django settings for RETINAL-AI backend.

Replaces FastAPI app factory (services/backend/src/main.py).
All env-var defaults mirror the old FastAPI config.py so nothing in .env needs changing.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Security ──────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "retinal-ai-django-insecure-secret-change-in-production-abc123xyz",
)
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
ALLOWED_HOSTS = ["*"]  # Nginx is the public edge; internal host is fine with wildcard

# ── Apps ──────────────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    # Django core (no ORM/admin needed — pure API service)
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "corsheaders",
    # Our API app
    "api.apps.ApiConfig",
]

# ── Middleware ─────────────────────────────────────────────────────────────────
# Order matters: GZip first, then CORS, then our custom timing middleware.
MIDDLEWARE = [
    "django.middleware.gzip.GZipMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "api.middleware.RequestIdTimingMiddleware",
]

# ── URL routing ───────────────────────────────────────────────────────────────
ROOT_URLCONF = "retinal_backend.urls"

# ── ASGI / WSGI ───────────────────────────────────────────────────────────────
ASGI_APPLICATION = "retinal_backend.asgi.application"
WSGI_APPLICATION = "retinal_backend.wsgi.application"

# ── CORS (replaces FastAPI CORSMiddleware) ────────────────────────────────────
_cors_origins_raw = os.environ.get("CORS_ORIGINS", "*")
if _cors_origins_raw.strip() == "*":
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS"]
CORS_ALLOW_HEADERS = ["*"]
CORS_EXPOSE_HEADERS = ["X-Request-Id", "X-Process-Time-Ms"]

# ── File upload limits (50 MB — Nginx enforces 50 MB, backend enforces 20 MB) ─
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024   # 50 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024   # 50 MB
DATA_UPLOAD_MAX_NUMBER_FIELDS = 20               # for batch endpoint

# ── Templates (unused — pure JSON API, but required by Django) ────────────────
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]

# ── Static files ──────────────────────────────────────────────────────────────
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# ── Django REST Framework ─────────────────────────────────────────────────────
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.JSONParser",
    ],
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ── Logging ───────────────────────────────────────────────────────────────────
_log_level = os.environ.get("LOG_LEVEL", "info").upper()
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{levelname}] {asctime} {name} — {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": _log_level,
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "WARNING",
            "propagate": False,
        },
        "django.server": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
