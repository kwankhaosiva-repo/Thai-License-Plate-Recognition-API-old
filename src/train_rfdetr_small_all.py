"""
src/train_rfdetr_small_all.py

Master sequential pipeline: trains ALL 4 RF-DETR-Small (Apache-2.0) models:
  1. Model 1  — Plate Detector    → weights/plate_detector_rfdetr_small.pt
  2. Model 2  — Components        → weights/component_detector_rfdetr_small.pt
  3. Model 3A — Char Box Detector → weights/character_box_detector_rfdetr_small.pt
  4. Lao      — Lao Plate Det.    → weights/plate_detector_lao_rfdetr_small.pt

Preserves all RT-DETR and RF-DETR-Base weights.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_rfdetr_small_plate      import train_rfdetr_small_plate
from train_rfdetr_small_components import train_rfdetr_small_components
from train_rfdetr_small_charbox    import train_rfdetr_small_charbox
from train_rfdetr_small_lao_plate  import train_rfdetr_small_lao_plate


STEPS = [
    ("Model 1 — Plate Detector (Small)",        train_rfdetr_small_plate),
    ("Model 2 — Component Detector (Small)",    train_rfdetr_small_components),
    ("Model 3A — Char Box Detector (Small)",    train_rfdetr_small_charbox),
    ("Lao — Lao Plate Detector (Small)",        train_rfdetr_small_lao_plate),
]


def main():
    total_start = time.time()
    print("=" * 80)
    print("🔥  RF-DETR-Small (Apache-2.0) — FULL TRAINING SEQUENCE")
    print("    Existing RT-DETR & RF-DETR-Base weights are 100% PRESERVED")
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
    print(f"🎉  RF-DETR-Small Training complete in {total_elapsed:.1f} minutes")
    print("-" * 80)
    for name, status in results.items():
        print(f"  {status}  —  {name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
