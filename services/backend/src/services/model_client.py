"""
ModelClient — async HTTP client for the internal model microservice.

Architecture:
  Backend Service  ──HTTP──▶  Model Service (/inference)
                               (GPU container)

The client is built on top of httpx (async) with:
  • Connection pooling (single shared AsyncClient)
  • Configurable timeouts
  • Automatic retries with exponential back-off on transient errors
  • Structured logging with per-request correlation IDs

This module owns the interface contract between the backend and model
services.  The model service schema lives in services/model/src/schemas.py;
we replicate the response fields we care about in TypedDicts here so that
the backend has no import dependency on the model service package at runtime.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Dict, List, TypedDict

import httpx

from services.backend.src.config import (
    MODEL_CONNECT_TIMEOUT,
    MODEL_MAX_RETRIES,
    MODEL_READ_TIMEOUT,
    MODEL_SERVICE_URL,
)

logger = logging.getLogger(__name__)

# ── Response TypedDicts (mirror model service schemas) ─────────────────────────


class InferenceResult(TypedDict):
    probabilities:   Dict[str, float]
    detected_labels: List[str]
    elapsed_ms:      float


# ── Shared async client ────────────────────────────────────────────────────────
# Instantiated once at import time; lifecycle managed by FastAPI lifespan.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError(
            "ModelClient not initialised. Call ModelClient.startup() first."
        )
    return _client


class ModelClient:
    """Namespace for model-service lifecycle and request methods."""

    # ── Lifecycle ──────────────────────────────────────────────────────── #

    @staticmethod
    async def startup() -> None:
        """Create the shared HTTP client.  Call once from FastAPI lifespan."""
        global _client
        _client = httpx.AsyncClient(
            base_url=MODEL_SERVICE_URL,
            timeout=httpx.Timeout(
                connect=MODEL_CONNECT_TIMEOUT,
                read=MODEL_READ_TIMEOUT,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
                keepalive_expiry=30,
            ),
            http2=False,  # keep simple; model service is internal
        )
        logger.info("ModelClient initialised — base_url=%s", MODEL_SERVICE_URL)

    @staticmethod
    async def shutdown() -> None:
        """Gracefully close the HTTP client.  Call once from FastAPI lifespan."""
        global _client
        if _client is not None:
            await _client.aclose()
            _client = None
            logger.info("ModelClient closed.")

    # ── Health check ───────────────────────────────────────────────────── #

    @staticmethod
    async def is_reachable() -> bool:
        """
        Ping the model service health endpoint.
        Returns False instead of raising on any network error.
        """
        try:
            resp = await _get_client().get("/health", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Inference ──────────────────────────────────────────────────────── #

    @staticmethod
    async def infer(
        image_bytes: bytes,
        threshold: float = 0.5,
    ) -> InferenceResult:
        """
        Call POST /inference on the model service and return the parsed result.

        Args:
            image_bytes: Raw bytes of a JPEG/PNG/BMP image.
            threshold:   Disease detection threshold forwarded to the model service.

        Returns:
            InferenceResult TypedDict with probabilities, detected_labels, elapsed_ms.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx from the model service.
            httpx.RequestError:    On network-level failures after all retries.
        """
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload   = {"image_b64": image_b64, "threshold": threshold}

        last_exc: Exception | None = None
        for attempt in range(1, MODEL_MAX_RETRIES + 2):
            try:
                t0   = time.perf_counter()
                resp = await _get_client().post("/inference", json=payload)
                resp.raise_for_status()
                elapsed = (time.perf_counter() - t0) * 1000

                data = resp.json()
                logger.debug(
                    "ModelClient.infer | attempt=%d | http_ms=%.1f "
                    "| model_ms=%.1f | detected=%d",
                    attempt,
                    elapsed,
                    data.get("elapsed_ms", 0),
                    len(data.get("detected_labels", [])),
                )
                return InferenceResult(
                    probabilities=data["probabilities"],
                    detected_labels=data["detected_labels"],
                    elapsed_ms=data["elapsed_ms"],
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt <= MODEL_MAX_RETRIES:
                    wait = 0.5 * (2 ** (attempt - 1))   # 0.5 s, 1 s, 2 s …
                    logger.warning(
                        "ModelClient.infer | transient error (attempt %d/%d) | "
                        "waiting %.1fs | %s",
                        attempt,
                        MODEL_MAX_RETRIES + 1,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
            except httpx.HTTPStatusError as exc:
                # 4xx / 5xx from model service — do not retry, propagate directly
                logger.error(
                    "ModelClient.infer | model service returned %d: %s",
                    exc.response.status_code,
                    exc.response.text[:200],
                )
                raise

        # All retries exhausted
        raise httpx.RequestError(
            f"Model service unreachable after {MODEL_MAX_RETRIES + 1} attempts: {last_exc}"
        ) from last_exc
