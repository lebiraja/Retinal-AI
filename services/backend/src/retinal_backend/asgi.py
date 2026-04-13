"""
ASGI entry-point — used by Gunicorn + UvicornWorker in production.

Replaces:  uvicorn services.backend.src.main:app
Now runs:  gunicorn retinal_backend.asgi:application -k uvicorn.workers.UvicornWorker
"""
import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retinal_backend.settings")

application = get_asgi_application()
