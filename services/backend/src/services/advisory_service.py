"""
AdvisoryService — converts raw model predictions into a structured risk assessment.

Risk Level Logic
----------------
┌───────────────────────────────────────────────────┬──────────────┐
│ Condition                                         │ Risk Level   │
├───────────────────────────────────────────────────┼──────────────┤
│ No diseases detected                              │ LOW          │
│ 1–2 diseases, none high-risk                      │ MODERATE     │
│ 3+ diseases  OR  exactly 1 high-risk disease      │ HIGH         │
│ 2+ high-risk diseases                             │ CRITICAL     │
└───────────────────────────────────────────────────┴──────────────┘

All advisory text is informational ONLY — this is not a diagnostic system.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

from services.backend.src.config import DISEASE_FULL_NAMES, HIGH_RISK_DISEASES


# ── Advisory text bank ─────────────────────────────────────────────────────────

_ADVISORY: dict[str, str] = {
    "LOW": (
        "No significant retinal abnormalities were detected in this image. "
        "While this is reassuring, it does not rule out early-stage eye disease. "
        "An annual comprehensive eye examination by a qualified ophthalmologist "
        "is recommended to maintain long-term ocular health."
    ),
    "MODERATE": (
        "Possible signs of retinal changes were identified in this image. "
        "This result is NOT a medical diagnosis. "
        "We recommend scheduling a consultation with a qualified ophthalmologist "
        "for a comprehensive dilated fundus examination within the next few weeks."
    ),
    "HIGH": (
        "Signs consistent with significant retinal pathology were detected. "
        "This result is NOT a medical diagnosis. "
        "Prompt evaluation by a certified ophthalmologist is strongly recommended — "
        "ideally within the next few days."
    ),
    "CRITICAL": (
        "Indicators associated with potentially sight-threatening retinal conditions "
        "were identified. This result is NOT a medical diagnosis. "
        "Immediate consultation with an ophthalmologist or emergency eye care "
        "service is strongly advised — do not delay."
    ),
}


# ── Return type ────────────────────────────────────────────────────────────────

class AdvisoryResult(TypedDict):
    disease_risk:            bool
    predictions:             Dict[str, float]
    detected_diseases:       List[str]
    detected_diseases_full:  List[str]
    num_detected:            int
    top_prediction:          Optional[str]
    confidence:              float
    risk_level:              str
    advisory:                str


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_advisory(predictions: Dict[str, float]) -> AdvisoryResult:
    """
    Convert a {label: probability} dict (diseases above threshold only) into
    a fully structured advisory result.

    Args:
        predictions: Filtered dict from the model service (only probs >= threshold).

    Returns:
        Populated AdvisoryResult TypedDict.
    """
    detected_labels: List[str] = list(predictions.keys())
    num_detected:    int       = len(detected_labels)
    disease_risk:    bool      = num_detected > 0

    # Highest-confidence disease label (None if nothing detected)
    top_prediction: Optional[str] = (
        max(predictions, key=predictions.__getitem__) if predictions else None
    )

    # Mean probability across detected diseases
    confidence: float = (
        round(float(sum(predictions.values()) / num_detected), 4)
        if predictions
        else 0.0
    )

    # Risk level calculation
    high_risk_detected = [d for d in detected_labels if d in HIGH_RISK_DISEASES]
    num_high_risk       = len(high_risk_detected)

    if num_detected == 0:
        risk_level = "LOW"
    elif num_high_risk >= 2:
        risk_level = "CRITICAL"
    elif num_high_risk == 1 or num_detected >= 3:
        risk_level = "HIGH"
    else:
        risk_level = "MODERATE"

    # Advisory text — prepend specific high-risk disease names for HIGH/CRITICAL
    advisory = _ADVISORY[risk_level]
    if high_risk_detected:
        full_names_str = ", ".join(
            DISEASE_FULL_NAMES.get(d, d) for d in high_risk_detected
        )
        advisory = f"Detected indicator(s): {full_names_str}. {advisory}"

    return AdvisoryResult(
        disease_risk=disease_risk,
        predictions=predictions,
        detected_diseases=detected_labels,
        detected_diseases_full=[
            DISEASE_FULL_NAMES.get(d, d) for d in detected_labels
        ],
        num_detected=num_detected,
        top_prediction=top_prediction,
        confidence=confidence,
        risk_level=risk_level,
        advisory=advisory,
    )
