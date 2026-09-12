"""
src/train_rtdetr_remaining.py

Sequential pipeline runner for training the remaining detection models on Apple Silicon MPS:
  1. Model 1 (Plate Detector): RT-DETR-L on LPR 2 - Polygon.yolov11_new
  2. Model 3A (Character Box Detector): RT-DETR-L on LPR 2 - Character Box Detection
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_rtdetr_plate import train_rtdetr_plate
from src.train_rtdetr_char_box import train_rtdetr_char_box


def main():
    total_start = time.time()
    print("=" * 80)
    print("🔥 MASTER ENTERPRISE TRAINING SEQUENCE: MODEL 1 & MODEL 3A (RT-DETR-L)")
    print("=" * 80)

    # 1. Train Model 1 (Plate Detector)
    t1_start = time.time()
    print("\n>>> [STEP 1/2] Training Model 1 (Plate Detector)...")
    train_rtdetr_plate()
    print(f"\n>>> [STEP 1/2] Finished Model 1 in {(time.time() - t1_start) / 60:.1f} minutes.")

    # 2. Train Model 3A (Character Box Detector)
    t2_start = time.time()
    print("\n>>> [STEP 2/2] Training Model 3A (Character Box Detector)...")
    train_rtdetr_char_box()
    print(f"\n>>> [STEP 2/2] Finished Model 3A in {(time.time() - t2_start) / 60:.1f} minutes.")

    print("\n" + "=" * 80)
    print(f"🎉 ALL ENTERPRISE MODELS SUCCESSFULLY TRAINED in {(time.time() - total_start) / 60:.1f} minutes!")
    print("=" * 80)


if __name__ == "__main__":
    main()
