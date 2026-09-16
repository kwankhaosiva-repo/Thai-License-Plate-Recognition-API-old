"""
src/train_dfine_nano_all.py

Master sequential training pipeline for D-FINE Nano (MIT License) models:
  1. Model 1  — Plate Detector    → weights/plate_detector_dfine_nano.pt & .onnx
  2. Model 2  — Components        → weights/component_detector_dfine_nano.pt & .onnx
  3. Model 3A — Char Box Detector → weights/character_box_detector_dfine_nano.pt & .onnx
  4. Lao      — Lao Plate Det.    → weights/plate_detector_lao_dfine_nano.pt & .onnx

Usage:
  # Train all 3 remaining models (skipping plate detector which is already trained):
  python src/train_dfine_nano_all.py --model components,charbox,lao_plate --epochs 30

  # Train everything:
  python src/train_dfine_nano_all.py --all --epochs 30
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_dfine_nano_plate      import train_dfine_nano_plate
from train_dfine_nano_components import train_dfine_nano_components
from train_dfine_nano_charbox    import train_dfine_nano_charbox
from train_dfine_nano_lao_plate  import train_dfine_nano_lao_plate


AVAILABLE_TASKS = {
    "plate":       ("Model 1 — Plate Detector (D-FINE Nano)",        train_dfine_nano_plate),
    "components":  ("Model 2 — Component Detector (D-FINE Nano)",    train_dfine_nano_components),
    "charbox":     ("Model 3A — Char Box Detector (D-FINE Nano)",    train_dfine_nano_charbox),
    "lao_plate":   ("Lao — Lao Plate Detector (D-FINE Nano)",        train_dfine_nano_lao_plate),
}


def main():
    parser = argparse.ArgumentParser(description="Master Training Pipeline for D-FINE Nano Object Detectors")
    parser.add_argument("--model", type=str, default="components,charbox,lao_plate",
                        help="Target models to train: 'components', 'charbox', 'lao_plate', 'plate', or comma-separated list, or 'all' (default: 'components,charbox,lao_plate')")
    parser.add_argument("--all", dest="train_all", action="store_true", help="Train all 4 models sequentially (including plate)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    parser.add_argument("--device", type=str, default="mps", help="Compute device: mps, cuda, or cpu (default: mps)")
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false", help="Skip automatic ONNX export")
    args = parser.parse_args()

    if args.train_all or args.model.strip().lower() == "all":
        selected_keys = list(AVAILABLE_TASKS.keys())
    else:
        raw_keys = [k.strip().lower() for k in args.model.split(",") if k.strip()]
        selected_keys = []
        for k in raw_keys:
            matched = False
            for valid_k in AVAILABLE_TASKS.keys():
                if k == valid_k or k in valid_k:
                    selected_keys.append(valid_k)
                    matched = True
                    break
            if not matched:
                print(f"⚠️ Unknown model key: '{k}'. Valid options: {list(AVAILABLE_TASKS.keys())}")

    if not selected_keys:
        print("❌ No valid tasks selected to train. Exiting.")
        return

    total_start = time.time()
    print("=" * 80)
    print("🔥  D-FINE Nano (MIT License) — TRAINING SEQUENCE")
    print(f"   Selected Tasks   : {', '.join(selected_keys)}")
    print(f"   Compute Device   : {args.device.upper()}")
    print(f"   Epochs / Task    : {args.epochs}")
    print(f"   Batch Size       : {args.batch}")
    print(f"   Image Resolution : {args.imgsz}x{args.imgsz}")
    print(f"   Early Stopping   : patience = {args.patience}")
    print(f"   Export ONNX      : {args.export_onnx}")
    print("   Existing weights : 100% PRESERVED")
    print("=" * 80)

    results = {}
    for i, key in enumerate(selected_keys, 1):
        name, train_fn = AVAILABLE_TASKS[key]
        t_start = time.time()
        print(f"\n>>> [{i}/{len(selected_keys)}] Starting: {name} ...")
        try:
            train_fn(
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                patience=args.patience,
                device=args.device,
                export_onnx=args.export_onnx,
            )
            elapsed = (time.time() - t_start) / 60
            results[name] = f"✅ Done in {elapsed:.1f} min"
            print(f">>> [{i}/{len(selected_keys)}] Finished {name} in {elapsed:.1f} minutes.")
        except Exception as e:
            results[name] = f"❌ FAILED: {e}"
            print(f">>> [{i}/{len(selected_keys)}] ⚠️  FAILED: {name} — {e}")
            print("    Continuing with remaining tasks...")

    total_elapsed = (time.time() - total_start) / 60
    print("\n" + "=" * 80)
    print(f"🎉  D-FINE Nano Training Finished in {total_elapsed:.1f} minutes")
    print("-" * 80)
    for name, status in results.items():
        print(f"  {status:<24}  —  {name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
