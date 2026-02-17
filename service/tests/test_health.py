"""Tests for health and root endpoints."""

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create test client with mocked camera manager."""
    with patch("main.camera_manager") as mock_cm:
        mock_cm.initialize = MagicMock(return_value=None)
        mock_cm.cleanup = MagicMock(return_value=None)
        mock_cm.get_active_count.return_value = 0

        # Need to make initialize/cleanup async
        import asyncio

        async def async_noop():
            pass

        mock_cm.initialize = MagicMock(side_effect=lambda: asyncio.coroutine(lambda: None)())
        mock_cm.cleanup = MagicMock(side_effect=lambda: asyncio.coroutine(lambda: None)())

        from main import app

        with TestClient(app) as client:
            yield client


def test_root_endpoint(test_client):
    """GET / should return service info."""
    resp = test_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "SentraAI"
    assert "endpoints" in data


def test_health_endpoint(test_client):
    """GET /api/health should return healthy status."""
    resp = test_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SentraAI"
    assert "version" in data
