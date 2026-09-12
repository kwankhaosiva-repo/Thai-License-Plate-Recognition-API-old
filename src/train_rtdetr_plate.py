"""
src/train_rtdetr_plate.py

Fine-tunes RT-DETR-L (Apache-2.0 License) for Model 1 Car Plate Detection.
Optimized for Apple Silicon MPS (Metal Performance Shaders).
Exports trained checkpoint to weights/plate_detector_rtdetr.pt upon completion.
"""

import sys
import shutil
from pathlib import Path
from ultralytics import RTDETR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new" / "data.yaml"
OUTPUT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
TARGET_WEIGHTS_PATH = OUTPUT_WEIGHTS_DIR / "plate_detector_rtdetr.pt"


def train_rtdetr_plate():
    print("=" * 70)
    print("🚀 Starting RT-DETR-L Enterprise Training for Model 1 (Plate Detection)")
    print(f"📁 Dataset YAML : {DATA_YAML}")
    print(f"🎯 Target Model : {TARGET_WEIGHTS_PATH}")
    print("=" * 70)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {DATA_YAML}")

    # Load base pretrained RT-DETR-L (Apache 2.0)
    print("\n[1/3] Loading RT-DETR-L pretrained foundation model...")
    model = RTDETR("rtdetr-l.pt")

    train_args = {
        "data": str(DATA_YAML),
        "epochs": 30,
        "patience": 10,
        "imgsz": 640,
        "batch": 4,
        "device": "mps",
        "workers": 2,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "project": str(PROJECT_ROOT / "runs" / "train_rtdetr"),
        "name": "plate",
        "exist_ok": True,
        "verbose": True,
        "plots": True,
        "save": True,
    }

    print("\n[2/3] Executing training loop on Apple Silicon MPS...")
    train_results = model.train(**train_args)

    best_pt = PROJECT_ROOT / "runs" / "train_rtdetr" / "plate" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = PROJECT_ROOT / "runs" / "train_rtdetr" / "plate" / "weights" / "last.pt"

    if best_pt.exists():
        print(f"\n[3/3] Training finished! Copying best weights to: {TARGET_WEIGHTS_PATH}")
        OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, TARGET_WEIGHTS_PATH)
        print(f"✅ Successfully exported enterprise model: {TARGET_WEIGHTS_PATH} ({TARGET_WEIGHTS_PATH.stat().st_size / 1e6:.2f} MB)")
    else:
        print("⚠️ Warning: Could not find checkpoint weights in runs/train_rtdetr/plate/weights/")

    # Validate
    print("\nEvaluating validation metrics...")
    val_model = RTDETR(str(TARGET_WEIGHTS_PATH if TARGET_WEIGHTS_PATH.exists() else best_pt))
    metrics = val_model.val(data=str(DATA_YAML), device="mps", batch=4)
    print(f"📊 Validation Results: mAP50 = {metrics.box.map50:.4f}, mAP50-95 = {metrics.box.map:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    train_rtdetr_plate()
