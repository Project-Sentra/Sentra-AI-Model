"""Simulated mode: camera only starts with a video picked from sample_videos/."""

import asyncio


def test_start_requires_listed_video(monkeypatch):
    from config import settings
    from services.camera_manager import CameraManager

    monkeypatch.setattr(settings, "CAMERA_MODE", "simulated")
    cm = CameraManager()
    asyncio.run(cm.initialize())
    assert all(v.lower().endswith((".mp4", ".avi", ".mov", ".mkv")) for v in cm.list_videos())

    # No pick, or a path outside sample_videos/ -> refused, nothing started
    assert not asyncio.run(cm.start_camera("entry_cam_01"))
    assert not asyncio.run(cm.start_camera("entry_cam_01", "../service/.env"))
    assert cm.get_camera("entry_cam_01").error_message == "Select a video to simulate"


def test_plate_fires_once_per_run(monkeypatch):
    from models.detector import DetectionResult
    import services.plate_detector as pd

    def fake_detect(frame, min_conf):
        r = DetectionResult()
        r.plate_text, r.plate_confidence = "CAG 5124", 0.9
        return r

    monkeypatch.setattr(pd, "detect_plate_in_frame", fake_detect)
    svc, seen = pd.PlateDetectorService(), set()
    events = [asyncio.run(svc.process_frame(None, "entry_cam_01", "entry", seen))[1] for _ in range(5)]
    assert sum(e is not None for e in events) == 1

    # New run (fresh set) reports it again
    assert asyncio.run(svc.process_frame(None, "entry_cam_01", "entry", set()))[1] is not None
