# Use lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for GCP Cloud Run & CPU inference optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    TZ="Asia/Bangkok" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    GCS_WEIGHTS_BUCKET="lpr-weight"

# Install system dependencies for OpenCV, image processing, and Thai fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    fonts-thai-tlwg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels first (slashes image size from ~3.5GB to ~180MB for fast Cloud Run cold start)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and web static assets.
# Weights: the 19 production files (~285 MB) are BAKED into the image — this
# removes the per-instance GCS download at boot (~50s cold-start penalty).
# .dockerignore whitelists exactly those files, so the ~6 GB of experimental
# checkpoints in weights/ never enter the build context. NOTE: deploying via
# `gcloud builds submit` uploads the local context straight to Cloud Build —
# NOTHING is pushed to GitHub (.gitignore already excludes weights/).
COPY src/ ./src/
COPY static/ ./static/
COPY weights/ ./weights/

# Expose Cloud Run default port
EXPOSE 8080

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:' + ('${PORT}' or '8080') + '/health')" || exit 1

# Download weights from GCS as a FALLBACK for any baked file that is missing
# (skips files already present), then start Uvicorn
CMD ["sh", "-c", "python3 src/download_weights.py && exec uvicorn src.api_server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]