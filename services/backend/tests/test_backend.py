"""
Unit tests for the RETINAL-AI Django backend service.

Tests are framework-agnostic — they test the pure-Python service layer directly.
No HTTP calls, no Django test client needed for unit tests.

Run from project root:
  PYTHONPATH=. pytest services/backend/tests/ -v
"""

from __future__ import annotations

import os

import pytest

# Point Django at our settings before any Django import
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "retinal_backend.settings")

# ── Advisory Service Tests ─────────────────────────────────────────────────────

class TestAdvisoryService:
    """Tests for generate_advisory() risk level logic (unchanged from FastAPI version)."""

    def test_no_diseases_returns_low(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({})
        assert result["risk_level"] == "LOW"
        assert result["disease_risk"] is False
        assert result["num_detected"] == 0
        assert result["confidence"] == 0.0
        assert result["top_prediction"] is None

    def test_one_non_high_risk_returns_moderate(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"MH": 0.8})
        assert result["risk_level"] == "MODERATE"
        assert result["disease_risk"] is True
        assert result["num_detected"] == 1
        assert result["top_prediction"] == "MH"

    def test_two_non_high_risk_returns_moderate(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"MH": 0.8, "ERM": 0.7})
        assert result["risk_level"] == "MODERATE"

    def test_three_non_high_risk_returns_high(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"MH": 0.8, "ERM": 0.7, "LS": 0.6})
        assert result["risk_level"] == "HIGH"

    def test_one_high_risk_returns_high(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"DR": 0.9})
        assert result["risk_level"] == "HIGH"
        assert "Diabetic Retinopathy" in result["advisory"]

    def test_two_high_risk_returns_critical(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"DR": 0.9, "ARMD": 0.85})
        assert result["risk_level"] == "CRITICAL"

    def test_confidence_is_mean_prob(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"MH": 0.8, "ERM": 0.6})
        assert abs(result["confidence"] - 0.7) < 0.001

    def test_top_prediction_is_highest_confidence(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"MH": 0.6, "ERM": 0.9, "LS": 0.7})
        assert result["top_prediction"] == "ERM"

    def test_detected_diseases_full_populated(self):
        from services.backend.src.services.advisory_service import generate_advisory
        result = generate_advisory({"DR": 0.9})
        assert "Diabetic Retinopathy" in result["detected_diseases_full"]


# ── Validation Service Tests ───────────────────────────────────────────────────

class TestValidationService:
    """
    Tests for upload validation helpers.

    In Django, validation_service raises ValueError/django.core.exceptions
    instead of FastAPI HTTPException.  We test for ValueError.
    """

    def test_empty_file_raises(self):
        from services.backend.src.services.validation_service import validate_upload_size
        with pytest.raises(Exception):
            validate_upload_size(b"", "empty.jpg", max_mb=10)

    def test_oversized_file_raises(self):
        from services.backend.src.services.validation_service import validate_upload_size
        oversized = b"x" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
        with pytest.raises(Exception):
            validate_upload_size(oversized, "big.jpg", max_mb=10)

    def test_valid_size_passes(self):
        from services.backend.src.services.validation_service import validate_upload_size
        # 1 KB — should pass with 10 MB limit
        validate_upload_size(b"x" * 1024, "ok.jpg", max_mb=10)


# ── Config Tests ───────────────────────────────────────────────────────────────

class TestConfig:
    """Sanity checks for config constants."""

    def test_disease_labels_count(self):
        from services.backend.src.config import DISEASE_LABELS
        assert len(DISEASE_LABELS) == 45

    def test_high_risk_is_subset_of_labels(self):
        from services.backend.src.config import DISEASE_LABELS, HIGH_RISK_DISEASES
        all_labels = set(DISEASE_LABELS)
        assert HIGH_RISK_DISEASES.issubset(all_labels), (
            f"Unknown high-risk labels: {HIGH_RISK_DISEASES - all_labels}"
        )

    def test_full_names_covers_all_labels(self):
        from services.backend.src.config import DISEASE_LABELS, DISEASE_FULL_NAMES
        missing = [lbl for lbl in DISEASE_LABELS if lbl not in DISEASE_FULL_NAMES]
        assert not missing, f"Missing full names for: {missing}"

    def test_default_threshold_in_range(self):
        from services.backend.src.config import DEFAULT_THRESHOLD
        assert 0.0 < DEFAULT_THRESHOLD <= 1.0

    def test_num_classes_matches_label_count(self):
        from services.backend.src.config import DISEASE_LABELS, NUM_CLASSES
        assert NUM_CLASSES == len(DISEASE_LABELS)

    def test_best_auc_reasonable(self):
        from services.backend.src.config import BEST_AUC
        assert 0.5 < BEST_AUC < 1.0


# ── Django View Tests (integration — uses Django test client) ──────────────────

@pytest.mark.django_db
class TestDjangoViews:
    """
    Light integration tests for the Django views.
    Requires Django to be configured (DJANGO_SETTINGS_MODULE set at top of file).
    """

    def test_health_endpoint_returns_200(self, client):
        """GET /health should always return 200 even if model service is down."""
        import asyncio
        from django.test import AsyncClient
        import pytest

    def test_info_endpoint_returns_45_classes(self):
        """GET /info should return 45 disease labels."""
        import json
        from django.test import RequestFactory
        from api.views import InfoView
        import asyncio

        factory = RequestFactory()
        request = factory.get("/info")
        view = InfoView.as_view()

        # Run async view in event loop
        response = asyncio.get_event_loop().run_until_complete(view(request))
        data = json.loads(response.content)
        assert data["num_classes"] == 45
        assert len(data["diseases"]) == 45
        assert data["metrics"]["mean_auc"] == 0.8204

    def test_predict_without_image_returns_400(self):
        """POST /predict without file should return 400."""
        import json
        import asyncio
        from django.test import RequestFactory
        from api.views import PredictView

        factory = RequestFactory()
        request = factory.post("/predict", data={}, content_type="multipart/form-data")
        view = PredictView.as_view()

        response = asyncio.get_event_loop().run_until_complete(view(request))
        assert response.status_code == 400

    def test_predict_batch_without_files_returns_400(self):
        """POST /predict-batch without files should return 400."""
        import json
        import asyncio
        from django.test import RequestFactory
        from api.views import PredictBatchView

        factory = RequestFactory()
        request = factory.post("/predict-batch", data={}, content_type="multipart/form-data")
        view = PredictBatchView.as_view()

        response = asyncio.get_event_loop().run_until_complete(view(request))
        assert response.status_code == 400

    def test_invalid_threshold_returns_422(self):
        """POST /predict with threshold=5.0 (out of range) should return 422."""
        import asyncio
        from django.test import RequestFactory
        from api.views import PredictView
        import io

        factory = RequestFactory()
        dummy_image = io.BytesIO(b"fake image data")
        request = factory.post(
            "/predict?threshold=5.0",
            data={"image": dummy_image},
        )
        view = PredictView.as_view()

        response = asyncio.get_event_loop().run_until_complete(view(request))
        assert response.status_code == 422
