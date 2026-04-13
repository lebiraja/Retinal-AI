"""
WSGI entry-point — kept for completeness / local dev with runserver.
Production uses asgi.py via Gunicorn + UvicornWorker.
"""
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retinal_backend.settings")

application = get_wsgi_application()
