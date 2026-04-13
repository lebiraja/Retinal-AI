"""
Django AppConfig for the 'api' application.

AppConfig.ready() replaces FastAPI's @app.on_event("startup") lifespan hook.
ModelClient uses lazy async initialisation so we only log here — no blocking
async calls in ready() which is synchronous.
"""
import logging
import os

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ApiConfig(AppConfig):
    name = "api"
    verbose_name = "Retinal AI — Prediction API"

    def ready(self) -> None:
        model_url = os.environ.get("MODEL_SERVICE_URL", "http://model-service:8001")
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
        logger.info("=" * 60)
        logger.info("RETINAL-AI Django backend initialising")
        logger.info("  Model Service  : %s", model_url)
        logger.info("  Ollama VLM     : %s", ollama_url)
        logger.info("  Django version : %s", self._django_version())
        logger.info("=" * 60)

    @staticmethod
    def _django_version() -> str:
        import django
        return django.get_version()
