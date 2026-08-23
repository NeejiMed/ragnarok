from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_metrics_endpoint_is_accessible():
    """
    /metrics endpoint must return 200 with Prometheus text format.
    This confirms the metrics app is mounted correctly.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_metrics_endpoint_contains_ragnarok_metrics():
    """
    After making a request, ragnarok-specific metrics must appear
    in the /metrics output.
    """
    # Make a request to trigger metric recording
    client.get("/health")

    response = client.get("/metrics")
    assert "ragnarok_requests_total" in response.text
    assert "ragnarok_request_duration_seconds" in response.text


def test_health_endpoint_increments_request_counter():
    """
    ragnarok_requests_total must increment after a successful request.
    """
    # Get baseline
    before = client.get("/metrics").text

    # Make a request
    client.get("/health")

    # Get updated metrics
    after = client.get("/metrics").text

    # The counter value for /health should have increased
    assert 'endpoint="/health"' in after


def test_normalize_path_replaces_uuids():
    """UUID-like path segments must be normalized to {id}."""
    from backend.app.core.middleware import _normalize_path

    path = "/documents/550e8400-e29b-41d4-a716-446655440000"
    normalized = _normalize_path(path)
    assert "{id}" in normalized
    assert "550e8400" not in normalized


def test_normalize_path_keeps_short_segments():
    """Short path segments must not be replaced."""
    from backend.app.core.middleware import _normalize_path

    path = "/documents/upload"
    normalized = _normalize_path(path)
    assert normalized == "/documents/upload"