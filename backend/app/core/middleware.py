import time 
from collections.abc import Callable

from fastapi import Request, Response

from backend.app.core.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    REQUEST_IN_PROGRESS
)

async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
    """
    FastAPI middleware to record Prometheus metrics for every request.
    Wraps each request: records start time, increments in-progress gauge, 
    calls the actual handler and then records duration and status.
    """
    method = request.method
    # Normalize endpoint, strip path parameters to avoid high cardinality explosion
    # For example, /documents/abc123 -> /documents/{id}
    endpoint = _normalize_path(request.url.path)

    REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
    except Exception as exc:
        status_code = "500"
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, http_status=status_code
        ).inc()
        raise exc  # Re-raise the exception after recording metrics
    finally:
        duration = time.perf_counter() - start_time
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    REQUEST_COUNT.labels(
        method=method, endpoint=endpoint, http_status=status_code
    ).inc()

    return response

def _normalize_path(path: str) -> str:
    """
    Normalizes URL paths to prevent high cardinality in Prometheus labels.
    High cardinality = too many unique label combinations = memory explosion.
    e.g. /documents/uuid-abc-123 would create a unique metric per document ID.
    We collapse these to /documents/{id} instead.
    """
    parts = path.split("/")
    normalized = []
    for part in parts:
        # If part looks like a UUID or a long hex string — replace with placeholder
        if len(part) > 20 or (len(part) == 36 and part.count("-") == 4):
            normalized.append("{id}")
        else:
            normalized.append(part)
    return "/".join(normalized)
    """
    Normalizes URL paths to prevent high cordinality in Prometheus labels.
    High cordinality = too many unique label combinations = memory explosion in Prometheus.
    e.g., /documents/uuid-abc-123 would create a unique metric per document ID, which is not useful for monitoring.
    We collapse these to /documents/{id} instead.
    """
    parts = path.split('/')
    normalized = []
    for part in parts:
        # if part looks like a UUID or a long hex string, replace it with a placeholder
        if len(part) > 20 or (len(part) == 36 and part.count('-') == 4):
            normalized.append("{id}")
        else:
            normalized.append(part)
    return '/'.join(normalized)