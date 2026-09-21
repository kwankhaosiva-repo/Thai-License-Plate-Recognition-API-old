"""
src/download_weights.py

Pre-startup sync script: Downloads production AI model weights and JSON mappings
from Google Cloud Storage (GCS) bucket into weights/ directory before FastAPI boots.
Ensures zero cold-start crashes on serverless container environments (Cloud Run).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
BUCKET_NAME = os.environ.get("GCS_WEIGHTS_BUCKET", "lpr-weight")


def sync_weights_from_gcs():
    """Download all objects from GCS bucket into weights/ if missing or 0 bytes."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # If running locally without GCP credentials and weights exist, don't fail
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs())

        if not blobs:
            print(f"⚠️ [GCS Sync] Bucket '{BUCKET_NAME}' is empty or unreachable.")
            return

        print(f"📦 [GCS Sync] Syncing {len(blobs)} files from gs://{BUCKET_NAME} to {WEIGHTS_DIR} ...")
        synced_count = 0

        for blob in blobs:
            # Avoid directory placeholders if any
            if blob.name.endswith("/"):
                continue

            target_path = WEIGHTS_DIR / Path(blob.name).name
            if not target_path.exists() or target_path.stat().st_size == 0:
                print(f"  ⬇️ Downloading: {blob.name} ({blob.size / (1024*1024):.1f} MB) -> {target_path.name}")
                blob.download_to_filename(str(target_path))
                synced_count += 1
            else:
                # Already present
                pass

        print(f"✅ [GCS Sync] Completed. {synced_count} files downloaded, {len(blobs) - synced_count} already cached.")

    except Exception as e:
        print(f"ℹ️ [GCS Sync Note] {e}")
        # Count existing model files in weights/
        existing = list(WEIGHTS_DIR.glob("*.pt")) + list(WEIGHTS_DIR.glob("*.pth"))
        print(f"ℹ️ [GCS Sync Note] Found {len(existing)} existing weight files in local weights/ directory.")


if __name__ == "__main__":
    sync_weights_from_gcs()
