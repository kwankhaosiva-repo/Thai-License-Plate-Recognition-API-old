"""
src/train_rfdetr_all.py

Master sequential pipeline: trains ALL RF-DETR-Base (Apache-2.0) models.
Mirrors the original train_rtdetr_remaining.py but for RF-DETR.

Models trained (same datasets as original RT-DETR):
  1. Model 1  — Plate Detector    → weights/plate_detector_rfdetr.pt
  2. Model 2  — Components        → weights/component_detector_rfdetr.pt
  3. Model 3A — Char Box Detector → weights/character_box_detector_rfdetr.pt
  4. Lao      — Lao Plate Det.    → weights/plate_detector_lao_rfdetr.pt

Old RT-DETR weights are PRESERVED — new files use different names.

Dataset confirmation (same as original RT-DETR training):
  Model 1  : datasets/Thai/LPR 2 - Polygon.yolov11_new/
  Model 2  : datasets/Thai/LPR 2 - Charactor Detection.yolov11/
  Model 3A : datasets/Thai/LPR 2 - Character Box Detection.yolov11/
  Lao      : datasets/Lao/laos plate.v3i.yolov11/
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_rfdetr_plate      import train_rfdetr_plate
from train_rfdetr_components import train_rfdetr_components
from train_rfdetr_charbox    import train_rfdetr_charbox
from train_rfdetr_lao_plate  import train_rfdetr_lao_plate


STEPS = [
    ("Model 1 — Plate Detector",        train_rfdetr_plate),
    ("Model 2 — Component Detector",    train_rfdetr_components),
    ("Model 3A — Char Box Detector",    train_rfdetr_charbox),
    ("Lao — Lao Plate Detector",        train_rfdetr_lao_plate),
]


def main():
    total_start = time.time()
    print("=" * 80)
    print("🔥  RF-DETR-Base (Apache-2.0) — FULL TRAINING SEQUENCE")
    print("    All old RT-DETR weights are PRESERVED (different filenames)")
    print("=" * 80)

    results = {}
    for i, (name, fn) in enumerate(STEPS, 1):
        t_start = time.time()
        print(f"\n>>> [{i}/{len(STEPS)}] Training: {name} ...")
        try:
            fn()
            elapsed = (time.time() - t_start) / 60
            results[name] = f"✅ Done in {elapsed:.1f} min"
            print(f">>> [{i}/{len(STEPS)}] Finished {name} in {elapsed:.1f} minutes.")
        except Exception as e:
            results[name] = f"❌ FAILED: {e}"
            print(f">>> [{i}/{len(STEPS)}] ⚠️  FAILED: {name} — {e}")
            print("    Continuing with next model...")

    total_elapsed = (time.time() - total_start) / 60
    print("\n" + "=" * 80)
    print(f"🎉  Training complete in {total_elapsed:.1f} minutes")
    print("-" * 80)
    for name, status in results.items():
        print(f"  {status}  —  {name}")
    print("=" * 80)
    print("\nNext step → update api_server.py to support RF-DETR inference.")
    print("Or run individual scripts to retrain a specific model.")


if __name__ == "__main__":
    main()
