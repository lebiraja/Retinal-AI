"""
VLMService — Vision-language model integration via local Ollama.

Two-stage pipeline:

  Stage 1 · Gate (FAIL-CLOSED)
    Is this an eye / retinal image?
        → YES:  continue to model inference.
        → NO:   reject with HTTP 422.
        → ERROR: reject with HTTP 503 (do NOT proceed to inference).

  Stage 2 · Analysis (FAIL-OPEN)
    Image + EfficientNet-B4 output → rich, personalised clinical advisory.
    Falls back to static advisory text if the VLM call fails.

Model: local Ollama vision-capable model (default: gemma4:e2b).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Optional

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


# ── Custom exception for gate failures ─────────────────────────────────────────

class VLMUnavailableError(Exception):
    """Raised when the VLM gate cannot produce a definitive answer.

    This is a HARD failure — the caller must NOT proceed to model inference.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


# ── Singleton client ───────────────────────────────────────────────────────────

_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        from config import (
            OLLAMA_BASE_URL,
            OLLAMA_TIMEOUT_S,
        )
        _client = httpx.AsyncClient(
            base_url=OLLAMA_BASE_URL,
            timeout=OLLAMA_TIMEOUT_S,
        )
    return _client


def _b64_image(image_bytes: bytes) -> str:
    """Return base64 image data, downscaled for faster VLM processing."""
    from config import VLM_MAX_IMAGE_SIDE

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((VLM_MAX_IMAGE_SIDE, VLM_MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=85, optimize=True)
        payload = out.getvalue()
    except Exception:
        payload = image_bytes

    return base64.standard_b64encode(payload).decode("ascii")


# ── Stage 1 — eye gate (FAIL-CLOSED) ──────────────────────────────────────────

async def check_is_eye_image(image_bytes: bytes, content_type: str) -> bool:
    """
    Ask the VLM whether the uploaded image is an eye / retinal image.

    Returns:
        True  — looks like a retinal/eye image; proceed with model inference.
        False — not an eye image; caller should return a friendly rejection.

    Raises:
        VLMUnavailableError — the VLM could not produce a definitive answer
            after all retries. Caller MUST reject the request (HTTP 503).

    IMPORTANT: This function is FAIL-CLOSED. If it cannot reach the VLM or
    parse the answer, it raises instead of returning a fallback.
    """
    from config import (
        OLLAMA_MODEL,
        VLM_GATE_MAX_RETRIES,
        VLM_GATE_MAX_TOKENS,
        VLM_GATE_TIMEOUT_S,
    )

    b64 = _b64_image(image_bytes)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max(1, VLM_GATE_MAX_TOKENS),
        },
        "messages": [
            {
                "role": "user",
                "content": (
                    "Look at this image carefully. "
                    "Is it a photograph of an eye, retinal fundus image, OCT scan, "
                    "slit-lamp image, optic disc image, or any medical image of an eye? "
                    "Random photos, screenshots, documents, pixel art, memes, selfies, "
                    "and non-eye images should be answered NO. "
                    "Reply with EXACTLY one word: YES or NO."
                ),
                "images": [b64],
            }
        ],
    }

    last_error: str = "unknown"

    for attempt in range(1, VLM_GATE_MAX_RETRIES + 1):
        try:
            response = await _get_client().post(
                "/api/chat",
                json=payload,
                timeout=VLM_GATE_TIMEOUT_S,
            )
            response.raise_for_status()
            body = response.json()
            msg = body.get("message") or {}
            # Primary: read from 'content'. Fallback: read from 'thinking'
            # (gemma4 thinking models may place output in 'thinking' field)
            answer = (msg.get("content") or "").strip().upper()
            if not answer:
                thinking = (msg.get("thinking") or "").strip().upper()
                if thinking:
                    logger.info(
                        "[VLM gate] content was empty, extracted from thinking field"
                    )
                    answer = thinking

            if not answer:
                last_error = f"VLM returned empty content and thinking fields (attempt {attempt})"
                logger.warning("[VLM gate] %s | raw_message=%r", last_error, msg)
                if attempt < VLM_GATE_MAX_RETRIES:
                    await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
                continue

            is_eye = answer.startswith("YES")
            logger.info(
                "[VLM gate] answer=%r → is_eye=%s (attempt %d/%d)",
                answer, is_eye, attempt, VLM_GATE_MAX_RETRIES,
            )
            return is_eye

        except httpx.TimeoutException:
            last_error = f"VLM request timed out (attempt {attempt}/{VLM_GATE_MAX_RETRIES})"
            logger.warning("[VLM gate] %s", last_error)

        except httpx.HTTPStatusError as exc:
            last_error = (
                f"VLM returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]} (attempt {attempt})"
            )
            logger.error("[VLM gate] %s", last_error)

        except Exception as exc:
            last_error = f"VLM connection error: {exc!r} (attempt {attempt})"
            logger.error("[VLM gate] %s", last_error)

        # Exponential backoff before retry
        if attempt < VLM_GATE_MAX_RETRIES:
            wait = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
            await asyncio.sleep(wait)

    # All retries exhausted — FAIL CLOSED
    raise VLMUnavailableError(
        f"VLM eye-image gate failed after {VLM_GATE_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


# ── Stage 2 — analysis (FAIL-OPEN — static fallback is acceptable) ────────────

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

    NOTE: Unlike the gate, this function is FAIL-OPEN. If VLM is unavailable,
    the static advisory from AdvisoryService is perfectly adequate.
    """
    from config import (
        OLLAMA_MODEL,
        OLLAMA_TEMPERATURE,
        DISEASE_FULL_NAMES,
        VLM_ANALYSIS_MAX_TOKENS,
        VLM_ANALYSIS_TIMEOUT_S,
    )

    b64          = _b64_image(image_bytes)
    num_detected = len(disease_full_names)

    if disease_full_names:
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
        response = await _get_client().post(
            "/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_predict": VLM_ANALYSIS_MAX_TOKENS,
                },
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": [b64],
                    },
                ],
            },
            timeout=VLM_ANALYSIS_TIMEOUT_S,
        )
        response.raise_for_status()
        resp_body = response.json()
        msg = resp_body.get("message") or {}
        # Primary: content. Fallback: thinking field (gemma4 thinking models)
        analysis = (msg.get("content") or "").strip()
        if not analysis:
            analysis = (msg.get("thinking") or "").strip()
        logger.debug("VLM analysis produced %d chars", len(analysis))
        return analysis if analysis else None

    except Exception as exc:
        logger.warning("VLM analysis failed (using static fallback): %s", exc)
        return None
