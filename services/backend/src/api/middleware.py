"""
Custom Django middleware — mirrors FastAPI's _request_id_and_timing middleware.

Adds two response headers to every request:
  X-Request-Id     : echoes the incoming header or generates a new 12-char hex id
  X-Process-Time-Ms: wall-clock time for the full request/response cycle

Works with both sync and async Django views (Django 4.1+ adaptive middleware).
"""
import asyncio
import logging
import time
import uuid

logger = logging.getLogger(__name__)


class RequestIdTimingMiddleware:
    """Adaptive (sync + async) request-id / timing middleware."""

    async_capable = True
    sync_capable = True

    def __init__(self, get_response):
        self.get_response = get_response
        # Mark as coroutine so Django's ASGI handler keeps us async.
        if asyncio.iscoroutinefunction(self.get_response):
            self._is_coroutine = asyncio.coroutines._is_coroutine

    def __call__(self, request):
        if asyncio.iscoroutinefunction(self.get_response):
            return self._async_call(request)
        return self._sync_call(request)

    # ── sync path ─────────────────────────────────────────────────────────────
    def _sync_call(self, request):
        request_id = request.META.get("HTTP_X_REQUEST_ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()

        response = self.get_response(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response["X-Request-Id"] = request_id
        response["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.debug(
            "req_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            request_id,
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # ── async path ────────────────────────────────────────────────────────────
    async def _async_call(self, request):
        request_id = request.META.get("HTTP_X_REQUEST_ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()

        response = await self.get_response(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response["X-Request-Id"] = request_id
        response["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.debug(
            "req_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            request_id,
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
        )
        return response
