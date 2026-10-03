"""
Security tests: every camera/gate/detection endpoint of SentraAI must
require an admin/operator login, and operator actions must be forwarded
to the backend with the operator's token (not the service key).
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# EasyOCR pulls in PyTorch; it is not needed to test routing/auth.
sys.modules.setdefault("easyocr", MagicMock())

from main import app  # noqa: E402
from services import operator_auth  # noqa: E402
from services.camera_manager import mask_source  # noqa: E402
from services.parking_client import EntryResult, ParkingClient  # noqa: E402

client = TestClient(app)
AUTH = {"Authorization": "Bearer operator-token"}

PROTECTED = [
    ("get", "/api/cameras"),
    ("get", "/api/cameras/entry_cam_01"),
    ("post", "/api/cameras/entry_cam_01/start"),
    ("post", "/api/cameras/entry_cam_01/stop"),
    ("post", "/api/cameras/start-all"),
    ("post", "/api/cameras/stop-all"),
    ("post", "/api/entry"),
    ("post", "/api/exit"),
    ("post", "/api/detect/base64"),
    ("post", "/api/detect/image"),
]


@pytest.fixture(autouse=True)
def _clear_cache():
    operator_auth.clear_cache()
    yield
    operator_auth.clear_cache()


@pytest.mark.parametrize("method,path", PROTECTED)
def test_endpoints_reject_anonymous(method, path):
    resp = getattr(client, method)(path)
    assert resp.status_code == 401


@pytest.mark.parametrize("method,path", PROTECTED)
def test_endpoints_reject_non_operator(method, path):
    with patch.object(
        operator_auth, "verify_operator_token", AsyncMock(return_value=False)
    ):
        resp = getattr(client, method)(path, headers=AUTH)
    assert resp.status_code == 401


def test_health_stays_public():
    assert client.get("/api/health").status_code == 200


def test_websocket_rejects_missing_token():
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws"):
            pass


def test_websocket_rejects_invalid_token():
    with patch.object(
        operator_auth, "verify_operator_token", AsyncMock(return_value=False)
    ):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/ws?token=bad"):
                pass


def test_websocket_accepts_operator():
    with patch.object(
        operator_auth, "verify_operator_token", AsyncMock(return_value=True)
    ):
        with client.websocket_connect("/api/ws?token=good") as ws:
            assert ws.receive_json()["type"] == "cameras_list"


def test_entry_forwards_operator_token():
    """Operator-confirmed entry must be sent with the operator's token."""
    fake_entry = AsyncMock(
        return_value=EntryResult(success=True, message="ok", spot_name="A-01")
    )
    with patch.object(
        operator_auth, "verify_operator_token", AsyncMock(return_value=True)
    ), patch("routers.cameras.parking_client.vehicle_entry", fake_entry):
        resp = client.post(
            "/api/entry",
            json={"plate_number": "WP CAB-1234", "camera_id": "entry_cam_01"},
            headers=AUTH,
        )
    assert resp.status_code == 200
    fake_entry.assert_awaited_once_with("WP CAB-1234", operator_token="operator-token")


def test_parking_client_uses_operator_token_not_service_key():
    headers = ParkingClient._auth_headers("operator-token")
    assert headers == {"Authorization": "Bearer operator-token", "X-Service-Key": ""}
    assert ParkingClient._auth_headers(None) is None


def test_verify_operator_token_checks_role():
    def make_transport(role, status=200):
        def handler(request):
            assert request.headers["Authorization"] == "Bearer t"
            return httpx.Response(status, json={"user": {"role": role}})

        return httpx.MockTransport(handler)

    import asyncio

    real_client = httpx.AsyncClient

    for role, status, expected in [
        ("admin", 200, True),
        ("operator", 200, True),
        ("user", 200, False),
        ("admin", 401, False),
    ]:
        operator_auth.clear_cache()
        transport = make_transport(role, status)
        with patch(
            "services.operator_auth.httpx.AsyncClient",
            lambda **kw: real_client(transport=transport, **kw),
        ):
            assert asyncio.run(operator_auth.verify_operator_token("t")) is expected


def test_verify_operator_token_fails_closed_when_backend_down():
    import asyncio

    def handler(request):
        raise httpx.ConnectError("down")

    real_client = httpx.AsyncClient
    with patch(
        "services.operator_auth.httpx.AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    ):
        assert asyncio.run(operator_auth.verify_operator_token("t")) is False


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "rtsp://admin:secret@192.168.1.64:554/stream",
            "rtsp://***@192.168.1.64:554/stream",
        ),
        ("rtsp://192.168.1.64:554/stream", "rtsp://192.168.1.64:554/stream"),
        ("/videos/sample.mp4", "/videos/sample.mp4"),
        ("0", "0"),
    ],
)
def test_mask_source_hides_credentials(source, expected):
    assert mask_source(source) == expected
