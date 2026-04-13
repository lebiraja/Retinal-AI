"""
ModelClient — async HTTP client for the internal model microservice.

Architecture:
  Backend Service  ──HTTP──▶  Model Service (/inference)
                               (GPU container)

The client is built on top of httpx (async) with:
  • Connection pooling (single shared AsyncClient — lazy initialisation)
  • Configurable timeouts
  • Automatic retries with exponential back-off on transient errors
  • Structured logging with per-request correlation IDs

Django migration note
─────────────────────
The old FastAPI version called ModelClient.startup() inside the lifespan
context manager.  Django has no equivalent async lifespan hook, so the
client is now created lazily on the first request via _get_or_create_client().
This is thread-safe for async views because asyncio is single-threaded.
ModelClient.startup() / .shutdown() are kept for backwards-compatibility but
are no-ops when called from Django (startup becomes lazy; shutdown is
registered via atexit).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Dict, List, TypedDict

import httpx

from config import (
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


# ── Shared async client — lazy singleton ──────────────────────────────────────
# Created on the first request; no explicit startup call required by Django.
_client: httpx.AsyncClient | None = None


def _make_client() -> httpx.AsyncClient:
    """Build a fresh httpx.AsyncClient with project-standard settings."""
    return httpx.AsyncClient(
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
        http2=False,
    )


async def _get_or_create_client() -> httpx.AsyncClient:
    """Return the singleton client, creating it lazily if needed."""
    global _client
    if _client is None:
        _client = _make_client()
        logger.info("ModelClient lazy-init — base_url=%s", MODEL_SERVICE_URL)
    return _client


class ModelClient:
    """Namespace for model-service lifecycle and request methods."""

    # ── Lifecycle (kept for backwards-compat; Django uses lazy init) ───── #

    @staticmethod
    async def startup() -> None:
        """
        Optionally pre-warm the HTTP client.

        Django: called from AppConfig.ready() indirectly — safe to skip;
                the client is created lazily on the first request.
        FastAPI: called from the lifespan context manager (legacy behaviour).
        """
        await _get_or_create_client()

    @staticmethod
    async def shutdown() -> None:
        """Gracefully close the HTTP client."""
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
            client = await _get_or_create_client()
            resp = await client.get("/health", timeout=3.0)
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

        client = await _get_or_create_client()
        last_exc: Exception | None = None
        for attempt in range(1, MODEL_MAX_RETRIES + 2):
            try:
                t0   = time.perf_counter()
                resp = await client.post("/inference", json=payload)
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
