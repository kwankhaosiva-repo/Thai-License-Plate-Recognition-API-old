"""
src/train_yolo_province_thai.py

Trains a dedicated YOLO object detection model for Thai license plate provinces (77 classes)
using the newly balanced dataset: datasets/Thai/thai_province_yolo_balanced/data.yaml.

Outputs:
- weights/province_detector_yolo_thai.pt (Preserves existing weights/province_model.pth untouched)
"""

import argparse
import shutil
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "datasets" / "Thai" / "thai_province_yolo_balanced" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "weights"
TARGET_MODEL_PATH = OUTPUT_DIR / "province_detector_yolo_thai.pt"


def get_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_thai_province(epochs=30, batch=16, imgsz=416, model_base="yolo11n.pt"):
    print("=" * 70)
    print("🇹🇭 Training Thai Province YOLO Object Detector (77 Classes)")
    print(f"📁 Dataset: {DATA_YAML}")
    print(f"🎯 Target Weights: {TARGET_MODEL_PATH}")
    print(f"⚙️  Base Model: {model_base} | Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    print("=" * 70)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {DATA_YAML}. Run src/balance_thai_province_yolo_dataset.py first.")

    device = get_device()
    print(f"Using compute device: {device}")

    model = YOLO(model_base)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=2,
        project=str(PROJECT_ROOT / "runs" / "province_train"),
        name="thai_province_yolo",
        exist_ok=True,
        save=True,
        plots=True,
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
    )

    # Export best model
    best_candidate = PROJECT_ROOT / "runs" / "province_train" / "thai_province_yolo" / "weights" / "best.pt"
    if best_candidate.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_candidate, TARGET_MODEL_PATH)
        print(f"\n✅ Training complete! Best weights exported to:\n  -> {TARGET_MODEL_PATH}")
    else:
        print(f"\n⚠️ Could not find best.pt at {best_candidate}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained base model (yolo11n.pt / yolo11s.pt)")
    args = parser.parse_args()

    train_thai_province(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model_base=args.model,
    )
