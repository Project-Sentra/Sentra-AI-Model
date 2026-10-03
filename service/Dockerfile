# ============================================================
# Sentra AI Service — Optimized Docker Image
# Strategy:
#   - Stage 1 (builder): install all wheels
#   - Stage 2 (production): copy only the final wheels + app code
#
# Key optimizations vs original Dockerfile:
#   - PyTorch CPU-only (~300MB vs ~2.5GB for the CUDA build)
#   - opencv-python-headless (no libXext, libXrender, libSM needed)
#   - ONNX Runtime for fast INT8 model inference
#   - Target image size: ~900MB (down from ~3GB+)
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Build tools (needed to compile some Python extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Install PyTorch CPU-only FIRST (separate index URL keeps the full CUDA
#    build out of the image — saves ~1.5GB).
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Build wheels for the rest of the dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# ── Stage 2: Production ──────────────────────────────────────
FROM python:3.11-slim AS production

# Non-root user for security. It needs a home directory because EasyOCR
# downloads its recognition models to ~/.EasyOCR on first start
# (mount a volume there to keep them across restarts).
RUN groupadd -r sentra && useradd -r -m -g sentra sentra \
    && mkdir -p /home/sentra/.EasyOCR \
    && chown -R sentra:sentra /home/sentra

WORKDIR /app

# Runtime system deps for OpenCV + EasyOCR.
# libgl1 replaces libgl1-mesa-glx, which no longer exists in current Debian
# (ultralytics pulls in the non-headless opencv-python, which needs libGL).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy PyTorch CPU wheels from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Install the remaining dependency wheels
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy application source code
COPY --chown=sentra:sentra . .

# Copy quantized ONNX model files
#   These should already exist in the repo after running the export commands.
#   If missing, the service will fall back to the .pt files (slower but functional).
RUN mkdir -p ./models && chown sentra:sentra ./models

USER sentra

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:5001/api/health || exit 1

EXPOSE 5001

ENV PORT=5001
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
