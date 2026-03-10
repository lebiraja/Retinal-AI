"""
VLMService — Silent vision-language model integration via Featherless AI.

Two-stage pipeline (both stages are transparent to the user):

  Stage 1 · Gate
    Is this an eye / retinal image?
    → YES: continue to model inference.
    → NO:  raise a friendly HTTP 422 (no technical details exposed).

  Stage 2 · Analysis
    Image + EfficientNet-B4 output → rich, personalised clinical advisory.
    Falls back to static advisory text if the VLM call fails.

Model: moonshotai/Kimi-K2.5 via Featherless OpenAI-compatible endpoint.
API key / base URL are read from environment variables at call time.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ── Singleton client ───────────────────────────────────────────────────────────

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        from services.backend.src.config import (
            FEATHERLESS_API_KEY,
            FEATHERLESS_BASE_URL,
        )
        _client = AsyncOpenAI(
            api_key=FEATHERLESS_API_KEY,
            base_url=FEATHERLESS_BASE_URL,
            timeout=60.0,
        )
    return _client


def _b64_image_url(image_bytes: bytes, content_type: str) -> str:
    """Return a data-URI suitable for the OpenAI vision message format."""
    # Normalise to a valid MIME type for the data URI
    mime = content_type if content_type.startswith("image/") else "image/jpeg"
    b64  = base64.standard_b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


# ── Stage 1 — eye gate ─────────────────────────────────────────────────────────

async def check_is_eye_image(image_bytes: bytes, content_type: str) -> bool:
    """
    Ask the VLM whether the uploaded image is an eye / retinal image.

    Returns:
        True  — looks like a retinal/eye image; proceed with model inference.
        False — not an eye image; caller should return a friendly rejection.

    Fail-open: if the VLM call errors out the function returns True so the
    user is not blocked by an infrastructure failure.
    """
    from services.backend.src.config import (
        FEATHERLESS_MODEL,
        FEATHERLESS_TEMPERATURE,
    )

    data_url = _b64_image_url(image_bytes, content_type)

    try:
        response = await _get_client().chat.completions.create(
            model=FEATHERLESS_MODEL,
            temperature=FEATHERLESS_TEMPERATURE,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Does this image contain an eye or any part of an eye? "
                                "This includes: close-up eye photos, macro eye shots, "
                                "retinal fundus photographs, OCT scans, slit-lamp images, "
                                "optic disc images, any medical or non-medical eye photograph, "
                                "or any image where an eye is clearly the main subject. "
                                "Reply with exactly one word: YES or NO."
                            ),
                        },
                    ],
                }
            ],
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        is_eye = answer.startswith("YES")
        logger.debug("VLM eye-gate answer=%r → is_eye=%s", answer, is_eye)
        return is_eye

    except Exception as exc:
        # Infrastructure failure — fail-open so users are not blocked
        logger.warning("VLM eye-gate unavailable (failing open): %s", exc)
        return True


# ── Stage 2 — analysis ─────────────────────────────────────────────────────────

async def generate_analysis(
    image_bytes: bytes,
    content_type: str,
    disease_full_names: list[str],
    risk_level: str,
    probabilities: dict[str, float],
) -> Optional[str]:
    """
    Generate a personalised, image-grounded analysis by sending both the
    retinal image and the EfficientNet-B4 classification output to the VLM.

    Returns:
        A 2-3 sentence advisory string, or None if the VLM call fails
        (caller should then use the static fallback advisory).
    """
    from services.backend.src.config import (
        FEATHERLESS_MODEL,
        FEATHERLESS_TEMPERATURE,
        DISEASE_FULL_NAMES,
    )

    data_url     = _b64_image_url(image_bytes, content_type)
    num_detected = len(disease_full_names)

    if disease_full_names:
        # Build a concise summary of what the model found
        top_findings = sorted(probabilities.items(), key=lambda x: -x[1])[:5]
        prob_text    = ", ".join(
            f"{DISEASE_FULL_NAMES.get(k, k)} ({v:.0%})"
            for k, v in top_findings
        )
        model_summary = (
            f"The AI screening model detected {num_detected} potential finding(s): "
            f"{prob_text}. Overall risk level: {risk_level}."
        )
    else:
        model_summary = (
            "The AI screening model found no significant retinal abnormalities. "
            f"Risk level: {risk_level}."
        )

    system_prompt = (
        "You are an empathetic ophthalmic screening assistant. "
        "Your role is to communicate automated retinal screening results clearly "
        "and compassionately to patients. "
        "You do NOT make medical diagnoses. "
        "You always recommend consulting a qualified ophthalmologist."
    )

    user_prompt = (
        f"You are reviewing a retinal photograph with AI-assisted analysis results.\n\n"
        f"Model output: {model_summary}\n\n"
        f"Please write a 2–3 sentence personalised advisory for this patient. "
        f"Reference what you can observe in the image and what the model found. "
        f"Keep the language simple, warm, and non-technical. "
        f"End with a clear next-step recommendation. "
        f"Do not include any diagnostic conclusions. "
        f"Do not mention 'AI model', 'algorithm', or technical terms."
    )

    try:
        response = await _get_client().chat.completions.create(
            model=FEATHERLESS_MODEL,
            temperature=FEATHERLESS_TEMPERATURE,
            max_tokens=350,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        )
        analysis = (response.choices[0].message.content or "").strip()
        logger.debug("VLM analysis produced %d chars", len(analysis))
        return analysis if analysis else None

    except Exception as exc:
        logger.warning("VLM analysis failed (using static fallback): %s", exc)
        return None
