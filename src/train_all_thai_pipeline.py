"""
src/train_all_thai_pipeline.py

Master Training Orchestrator for Thai LPR Pipeline:
1. Model 1: Plate Detector (PicoDet-S, imgsz=320x320)
2. Model 2: Component Detector (plate_char vs province, PicoDet-S, imgsz=320x320)
3. Model 3A: Character Box Detector (PicoDet-S, imgsz=320x320)
4. Model 3B: Thai Character Classifier (MobileNetV2, 50 classes, robust augmentations)
5. Model 3B/Province: Grayscale Thai Province Classifier (ResNet18, 256x80)
6. Updates src/config.py with new model paths and runs verification
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_libreyolo_task import train_libreyolo_task
from src.train_character_classifier import train_character_classifier
from src.train_grayscale_province_thai import train_grayscale_thai


def train_thai_pipeline(epochs_det: int = 20, epochs_cls: int = 15, tag: str = "v6pico320", device: str = "mps"):
    print("=" * 80)
    print(f"🇹🇭 Starting Thai LPR Pipeline Retraining (Tag: {tag}, Device: {device})")
    print("=" * 80)
    start_total = time.time()
    weights_dir = PROJECT_ROOT / "weights"

    # ---------------------------------------------------------
    # Stage 1: Model 1 Plate Detector (PicoDet-S 320x320)
    # ---------------------------------------------------------
    m1_pt = weights_dir / f"plate_detector_picodet_s_{tag}.pt"
    if m1_pt.exists():
        print(f"\n[1/5] Model 1 already trained: {m1_pt.name} (skipping)")
    else:
        print("\n[1/5] Training Model 1: Plate Detector (PicoDet-S 320x320)...")
        train_libreyolo_task(
            task="plate",
            variant="picodet_s",
            imgsz=320,
            epochs=epochs_det,
            batch=8,
            aug="plate",
            device=device,
            tag=tag,
            overwrite=True,
        )

    # ---------------------------------------------------------
    # Stage 2: Model 2 Component Detector (PicoDet-S 320x320)
    # Uses LPR 2 - Charactor Detection.yolov11 (clean character bounding boxes)
    # ---------------------------------------------------------
    m2_pt = weights_dir / f"component_detector_picodet_s_{tag}.pt"
    if m2_pt.exists():
        print(f"\n[2/5] Model 2 already trained: {m2_pt.name} (skipping)")
    else:
        print("\n[2/5] Training Model 2: Component Detector (PicoDet-S 320x320)...")
        comp_dir = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Charactor Detection.yolov11"
        train_libreyolo_task(
            task="components",
            variant="picodet_s",
            imgsz=320,
            epochs=epochs_det,
            batch=8,
            aug="plate",
            data=str(comp_dir),
            device=device,
            tag=tag,
            overwrite=True,
        )

    # ---------------------------------------------------------
    # Stage 3: Model 3A Character Box Detector (PicoDet-S 320x320)
    # ---------------------------------------------------------
    m3a_pt = weights_dir / f"character_box_detector_picodet_s_{tag}.pt"
    if m3a_pt.exists():
        print(f"\n[3/5] Model 3A already trained: {m3a_pt.name} (skipping)")
    else:
        print("\n[3/5] Training Model 3A: Character Box Detector (PicoDet-S 320x320)...")
        train_libreyolo_task(
            task="charbox",
            variant="picodet_s",
            imgsz=320,
            epochs=epochs_det,
            batch=8,
            aug="plate",
            device=device,
            tag=tag,
            overwrite=True,
        )

    # ---------------------------------------------------------
    # Stage 4: Model 3B Character Classifier (MobileNetV2, 50 classes)
    # ---------------------------------------------------------
    m3b_pt = weights_dir / f"character_classifier_{tag}.pth"
    if m3b_pt.exists():
        print(f"\n[4/5] Model 3B already trained: {m3b_pt.name} (skipping)")
    else:
        print("\n[4/5] Training Model 3B: Thai Character Classifier...")
        train_character_classifier(
            epochs=epochs_cls,
            batch_size=64,
            tag=tag,
        )

    # ---------------------------------------------------------
    # Stage 5: Model 3B Province Classifier (ResNet18-Grayscale, 77 classes)
    # ---------------------------------------------------------
    prov_pt = weights_dir / f"province_model_grayscale_thai_{tag}.pth"
    if prov_pt.exists():
        print(f"\n[5/5] Model 3B/4 already trained: {prov_pt.name} (skipping)")
    else:
        print("\n[5/5] Training Model 3B/4: Thai Province Classifier...")
        train_grayscale_thai(
            epochs=epochs_cls,
            batch_size=32,
            backbone="resnet18",
            tag=tag,
        )

    elapsed_total = int(time.time() - start_total)
    print("\n" + "=" * 80)
    print(f"✅ Thai Pipeline Training Completed in {elapsed_total // 60}m {elapsed_total % 60}s!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Thai LPR Pipeline End-to-End")
    parser.add_argument("--epochs-det", type=int, default=20, help="Epochs for detection stages (default: 20)")
    parser.add_argument("--epochs-cls", type=int, default=15, help="Epochs for classification stages (default: 15)")
    parser.add_argument("--tag", type=str, default="v6pico320", help="Model version tag (default: v6pico320)")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    args = parser.parse_args()

    train_thai_pipeline(
        epochs_det=args.epochs_det,
        epochs_cls=args.epochs_cls,
        tag=args.tag,
        device=args.device,
    )
