"""
export_onnx.py — Sentra AI Model Quantization Export Script
============================================================

Run this ONCE on your local machine (not inside Docker) to convert the
heavy PyTorch .pt models into lightweight, INT8-quantized ONNX models.

Usage:
    python export_onnx.py

Prerequisites:
    pip install ultralytics onnx onnxslim onnxruntime

Output files (committed to the repo / baked into Docker image):
    service/yolov8n.onnx                        ← general vehicle detector
    models/license_plate_detector.onnx          ← custom plate detector

After running this script, verify with:
    python export_onnx.py --verify
"""

import argparse
import sys
from pathlib import Path

# ── Repo root paths ───────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
SERVICE_DIR = REPO_ROOT / "service"
MODELS_DIR = REPO_ROOT / "models"

YOLO_PT = SERVICE_DIR / "yolov8n.pt"
PLATE_PT = MODELS_DIR / "license_plate_detector.pt"

YOLO_ONNX = SERVICE_DIR / "yolov8n.onnx"
PLATE_ONNX = MODELS_DIR / "license_plate_detector.onnx"


def check_prerequisites():
    """Ensure required packages are installed."""
    missing = []
    for pkg in ["ultralytics", "onnx", "onnxruntime"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Install them first:\n")
        print("  pip install ultralytics onnx onnxslim onnxruntime\n")
        sys.exit(1)


def export_model(pt_path: Path, onnx_path: Path, label: str):
    """
    Export a single YOLOv8 .pt model to INT8 ONNX.

    Two-step process:
      1. Ultralytics exports the model to FP32 ONNX (int8 flag is NOT
         supported by the ONNX exporter — only TFLite/CoreML support it).
      2. onnxruntime.quantization.quantize_dynamic converts the FP32 ONNX
         to a dynamic INT8 ONNX in-place.
    """
    from ultralytics import YOLO
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not pt_path.exists():
        print(f"\n[SKIP] {label}: .pt file not found at {pt_path}")
        return False

    if onnx_path.exists():
        print(f"\n[SKIP] {label}: ONNX already exists at {onnx_path}")
        print("       Delete it and re-run to force re-export.")
        return True

    # ── Step 1: Export FP32 ONNX ──────────────────────────────
    print(f"\n[EXPORT] {label}")
    print(f"  Input : {pt_path}")
    print(f"  Output: {onnx_path}")
    print("  Step 1/2: Exporting FP32 ONNX …\n")

    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        simplify=True,
        dynamic=False,   # Static input shapes → faster inference
        opset=17,
    )

    # Ultralytics saves the .onnx next to the .pt by default
    fp32_path = pt_path.with_suffix(".onnx")
    if not fp32_path.exists():
        print(f"\n  [FAIL] FP32 ONNX not found at expected path: {fp32_path}")
        return False

    size_fp32_mb = fp32_path.stat().st_size / 1_048_576
    print(f"\n  FP32 export OK ({size_fp32_mb:.1f} MB) → {fp32_path}")

    # ── Step 2: Apply dynamic INT8 quantization ───────────────
    print(f"  Step 2/2: Applying dynamic INT8 quantization …")

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(onnx_path),
        weight_type=QuantType.QInt8,
    )

    # Clean up intermediate FP32 file if it's different from the output path
    if fp32_path != onnx_path and fp32_path.exists():
        fp32_path.unlink()
        print(f"  Removed intermediate FP32 file: {fp32_path.name}")

    if onnx_path.exists():
        size_int8_mb = onnx_path.stat().st_size / 1_048_576
        reduction = (1 - size_int8_mb / size_fp32_mb) * 100
        print(f"\n  [OK] {label} → INT8 ONNX saved ({size_int8_mb:.1f} MB, ~{reduction:.0f}% smaller)")
        return True
    else:
        print(f"\n  [FAIL] INT8 ONNX not found at: {onnx_path}")
        return False


def verify_models():
    """Quick inference sanity-check on both ONNX models."""
    import numpy as np
    import onnxruntime as ort

    print("\n── Verification ──────────────────────────────────────────")

    for label, onnx_path in [("Vehicle Detector", YOLO_ONNX), ("Plate Detector", PLATE_ONNX)]:
        if not onnx_path.exists():
            print(f"  [SKIP] {label}: {onnx_path} not found.")
            continue

        try:
            sess = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
            # Get expected input shape and run a dummy inference
            inp = sess.get_inputs()[0]
            shape = inp.shape  # e.g. [1, 3, 640, 640]
            # Replace dynamic dims with concrete sizes
            concrete_shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]
            dummy = np.random.rand(*concrete_shape).astype(np.float32)
            sess.run(None, {inp.name: dummy})
            print(f"  [OK] {label} — inference OK  (input shape: {shape})")
        except Exception as e:
            print(f"  [FAIL] {label} — {e}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Export Sentra AI YOLO models to INT8 ONNX.")
    parser.add_argument("--verify", action="store_true", help="Only run inference verification.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Sentra AI — ONNX INT8 Export Script")
    print("=" * 60)

    check_prerequisites()

    if args.verify:
        verify_models()
        return

    ok1 = export_model(YOLO_PT, YOLO_ONNX, "YOLOv8n Vehicle Detector")
    ok2 = export_model(PLATE_PT, PLATE_ONNX, "Custom Plate Detector")

    verify_models()

    print("=" * 60)
    if ok1 or ok2:
        print("  Export complete. Commit the .onnx files to your repo,")
        print("  or ensure they are present before building the Docker image.")
    else:
        print("  No models were exported (all either skipped or failed).")
        print("  Ensure .pt files exist in the correct paths.")
    print("=" * 60)


if __name__ == "__main__":
    main()
