"""
AdvisoryService — converts inference results into a structured risk assessment.

Risk level logic (from ARCHITECTURE.md):
  ┌──────────────────────────────────────────────┬──────────────┐
  │ Condition                                    │ Risk Level   │
  ├──────────────────────────────────────────────┼──────────────┤
  │ No diseases detected                         │ LOW          │
  │ 1–2 diseases, none high-risk                 │ MODERATE     │
  │ 3+ diseases  OR  any 1  high-risk disease    │ HIGH         │
  │ 2+ high-risk diseases                        │ CRITICAL     │
  └──────────────────────────────────────────────┴──────────────┘

This module produces informational text ONLY. It is NOT medical advice.
All advisory messages direct users toward a qualified ophthalmologist.
"""

from typing import Dict, List, Optional

from app.config import DISEASE_FULL_NAMES, HIGH_RISK_DISEASES

# ── Advisory text bank ────────────────────────────────────────────────────────

_ADVISORY: Dict[str, str] = {
    "LOW": (
        "No significant retinal abnormalities were detected in this image. "
        "This does not rule out eye disease. Regular annual eye examinations "
        "are still recommended to maintain ocular health."
    ),
    "MODERATE": (
        "Possible signs of retinal changes were identified. "
        "This is not a medical diagnosis. "
        "Please schedule a consultation with a qualified ophthalmologist "
        "for a comprehensive examination."
    ),
    "HIGH": (
        "Signs consistent with significant retinal pathology were detected. "
        "This is not a medical diagnosis. "
        "Prompt evaluation by a certified ophthalmologist is strongly recommended."
    ),
    "CRITICAL": (
        "The analysis identified indicators associated with potentially "
        "sight-threatening conditions. This is not a medical diagnosis. "
        "Immediate consultation with an ophthalmologist or emergency eye care "
        "service is strongly advised."
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────

def generate_advisory(predictions: Dict[str, float]) -> Dict:
    """
    Convert a {label: probability} dict (above-threshold diseases only) into
    a full structured advisory result.

    Args:
        predictions: Filtered dict from inference_service (prob >= threshold).

    Returns:
        {
            "disease_risk"           : bool,
            "predictions"            : { label: prob },
            "detected_diseases"      : [short labels],
            "detected_diseases_full" : [full names],
            "num_detected"           : int,
            "top_prediction"         : str | None,
            "confidence"             : float,   # mean prob of detected diseases
            "risk_level"             : "LOW|MODERATE|HIGH|CRITICAL",
            "advisory"               : str,
        }
    """
    detected_labels: List[str] = list(predictions.keys())
    num_detected:    int       = len(detected_labels)
    disease_risk:    bool      = num_detected > 0

    # ── Top prediction ────────────────────────────────────────────────── #
    top_prediction: Optional[str] = (
        max(predictions, key=predictions.__getitem__) if predictions else None
    )

    # ── Confidence = mean probability across ALL detected diseases ────── #
    # (matches BACKEND.md spec — not just the single top disease)
    confidence: float = (
        round(float(sum(predictions.values()) / num_detected), 4)
        if predictions else 0.0
    )

    # ── Risk level ────────────────────────────────────────────────────── #
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

    # ── Advisory text ─────────────────────────────────────────────────── #
    advisory = _ADVISORY[risk_level]

    # Prepend detected high-risk disease names for HIGH / CRITICAL
    if high_risk_detected:
        full_names = [DISEASE_FULL_NAMES.get(d, d) for d in high_risk_detected]
        advisory = f"Detected indicator(s): {', '.join(full_names)}. " + advisory

    return {
        "disease_risk":           disease_risk,
        "predictions":            predictions,
        "detected_diseases":      detected_labels,
        "detected_diseases_full": [DISEASE_FULL_NAMES.get(d, d) for d in detected_labels],
        "num_detected":           num_detected,
        "top_prediction":         top_prediction,
        "confidence":             confidence,
        "risk_level":             risk_level,
        "advisory":               advisory,
    }
