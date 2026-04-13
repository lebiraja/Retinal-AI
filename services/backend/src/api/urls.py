"""
URL patterns for the api application.

Nginx strips /api/ before forwarding here, so routes below map 1-to-1 with
the old FastAPI router (no /api/ prefix needed).
"""
from django.urls import path

from . import views

urlpatterns = [
    # Prediction endpoints
    path("predict", views.PredictView.as_view(), name="predict"),
    path("predict-batch", views.PredictBatchView.as_view(), name="predict-batch"),

    # Utility endpoints
    path("health", views.HealthView.as_view(), name="health"),
    path("info", views.InfoView.as_view(), name="info"),
]
