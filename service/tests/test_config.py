"""Tests for configuration module."""

import os


def test_settings_defaults():
    """Default settings should have expected values."""
    from config import Settings

    s = Settings()
    assert s.PORT == int(os.getenv("PORT", "5001"))
    assert s.HOST == os.getenv("HOST", "0.0.0.0")
    assert s.CAMERA_MODE in ("simulated", "live")
    assert s.MIN_CONFIDENCE > 0
    assert s.MIN_CONFIDENCE <= 1.0
    assert s.DETECTION_COOLDOWN >= 0
    assert s.FRAME_SKIP >= 1
    assert s.JPEG_QUALITY >= 1
    assert s.JPEG_QUALITY <= 100


def test_settings_model_paths():
    """Model paths should be set."""
    from config import Settings

    s = Settings()
    assert "yolov8n.pt" in s.YOLO_MODEL
    assert "license_plate_detector.pt" in s.PLATE_DETECTOR_MODEL


def test_settings_parking_api_url():
    """Parking API URL should have a default."""
    from config import Settings

    s = Settings()
    assert s.PARKING_API_URL.startswith("http")
