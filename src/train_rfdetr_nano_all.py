"""
src/train_rfdetr_nano_all.py

Master sequential training pipeline for all 4 RF-DETR-Nano (Apache-2.0) object detector models:
  1. Model 1  — Plate Detector    → weights/plate_detector_rfdetr_nano.pt
  2. Model 2  — Components        → weights/component_detector_rfdetr_nano.pt
  3. Model 3A — Char Box Detector → weights/character_box_detector_rfdetr_nano.pt
  4. Lao      — Lao Plate Det.    → weights/plate_detector_lao_rfdetr_nano.pt

Usage:
  python src/train_rfdetr_nano_all.py --all
  python src/train_rfdetr_nano_all.py --model components,charbox --epochs 30
  python src/train_rfdetr_nano_all.py --model plate --epochs 35 --export-onnx
"""

import argparse
import sys
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_rfdetr_nano_plate      import train_rfdetr_nano_plate
from train_rfdetr_nano_components import train_rfdetr_nano_components
from train_rfdetr_nano_charbox    import train_rfdetr_nano_charbox
from train_rfdetr_nano_lao_plate  import train_rfdetr_nano_lao_plate


AVAILABLE_TASKS = {
    "plate":       ("Model 1 — Plate Detector (Nano)",        train_rfdetr_nano_plate),
    "components":  ("Model 2 — Component Detector (Nano)",    train_rfdetr_nano_components),
    "charbox":     ("Model 3A — Char Box Detector (Nano)",    train_rfdetr_nano_charbox),
    "lao_plate":   ("Lao — Lao Plate Detector (Nano)",        train_rfdetr_nano_lao_plate),
}


def main():
    parser = argparse.ArgumentParser(description="Master Training Pipeline for RF-DETR-Nano Object Detectors")
    parser.add_argument("--model", type=str, default="all",
                        help="Target models to train: 'plate', 'components', 'charbox', 'lao_plate', or comma-separated list, or 'all'")
    parser.add_argument("--all", dest="train_all", action="store_true", help="Train all 4 models sequentially")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--device", type=str, default=None, help="Compute device: mps, cuda, or cpu (default: auto)")
    parser.add_argument("--export-onnx", action="store_true", help="Export trained models to standalone ONNX")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

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
    print("🔥  RF-DETR-Nano (Apache-2.0) — TRAINING SEQUENCE")
    print(f"   Selected Tasks   : {', '.join(selected_keys)}")
    print(f"   Compute Device   : {device.upper()}")
    print(f"   Epochs / Task    : {args.epochs}")
    print(f"   Batch Size       : {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
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
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum,
                lr=args.lr,
                patience=args.patience,
                device=device,
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
    print(f"🎉  RF-DETR-Nano Training Finished in {total_elapsed:.1f} minutes")
    print("-" * 80)
    for name, status in results.items():
        print(f"  {status:<24}  —  {name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
