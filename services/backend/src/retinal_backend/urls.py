"""
Root URL configuration for RETINAL-AI Django backend.

Nginx strips the /api/ prefix before forwarding requests here, so all
routes are registered without that prefix — matching FastAPI behaviour exactly.

Public URL (via Nginx) → Internal Django URL
  POST /api/predict        → POST /predict
  POST /api/predict-batch  → POST /predict-batch
  GET  /api/health         → GET  /health
  GET  /api/info           → GET  /info
"""
from django.urls import path, include

urlpatterns = [
    path("", include("api.urls")),
]
