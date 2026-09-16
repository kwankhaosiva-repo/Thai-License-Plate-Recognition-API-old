"""
src/train_rfdetr_nano_charbox.py

Fine-tunes RF-DETR-Nano (Apache-2.0 License) for Model 3A Character Box Detection.
Classes: char / Each Thai Charactor car plate (1 class).
Saves weights to weights/character_box_detector_rfdetr_nano.pt.
"""

import argparse
import shutil
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR         = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Character Box Detection.yolov11"
DATASET_YAML        = DATASET_DIR / "data.yaml"
OUTPUT_WEIGHTS_DIR  = PROJECT_ROOT / "weights"
TARGET_WEIGHTS_PATH = OUTPUT_WEIGHTS_DIR / "character_box_detector_rfdetr_nano.pt"
TARGET_ONNX_PATH    = OUTPUT_WEIGHTS_DIR / "character_box_detector_rfdetr_nano.onnx"
RUNS_DIR            = PROJECT_ROOT / "runs" / "train_rfdetr_nano" / "charbox"


def train_rfdetr_nano_charbox(epochs=30, batch_size=4, grad_accum_steps=2, lr=1e-4, patience=10, device=None, export_onnx=False):
    print("=" * 70)
    print("🚀 RF-DETR-Nano Training — Model 3A (Character Box Detection)")
    print(f"   License   : Apache-2.0 ✅")
    print(f"   Classes   : char (1 class)")
    print(f"   Dataset   : {DATASET_DIR.name}")
    print(f"   Config    : {DATASET_YAML}")
    print(f"   Output    : {TARGET_WEIGHTS_PATH}")
    print(f"   Epochs    : {epochs} | Batch: {batch_size} (accum: {grad_accum_steps}) | Patience: {patience}")
    print("=" * 70)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATASET_YAML}")

    from rfdetr import RFDETRNano

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n[1/3] Loading RF-DETR-Nano pretrained model (Apache-2.0) on {device}...")
    model = RFDETRNano()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] Starting fine-tuning...")
    model.train(
        dataset_dir=str(DATASET_DIR),
        dataset_file="yolo",
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        multi_scale=False,
        lr=lr,
        lr_encoder=lr * 0.1,
        output_dir=str(RUNS_DIR),
        use_ema=True,
        checkpoint_interval=1,
        early_stopping=True,
        early_stopping_patience=patience,
        accelerator=device,
        num_workers=2,
    )

    best_ckpt = None
    for name in [
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "checkpoint_best.pth",
        "checkpoint_best.pt",
        "best.pt",
    ]:
        candidate = RUNS_DIR / name
        if candidate.exists():
            best_ckpt = candidate
            break
    if best_ckpt is None:
        pths = sorted(RUNS_DIR.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if pths:
            best_ckpt = pths[0]

    if best_ckpt and best_ckpt.exists():
        OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, TARGET_WEIGHTS_PATH)
        size_mb = TARGET_WEIGHTS_PATH.stat().st_size / 1e6
        print(f"\n[3/3] ✅ Saved best weights → {TARGET_WEIGHTS_PATH} ({size_mb:.1f} MB)")

        if export_onnx:
            print(f"Exporting to ONNX → {TARGET_ONNX_PATH}...")
            try:
                trained_model = RFDETRNano.from_checkpoint(str(TARGET_WEIGHTS_PATH), trust_checkpoint=True)
                exp_path = trained_model.export(output_dir=str(RUNS_DIR), format="onnx", opset_version=17)
                if Path(exp_path).exists():
                    shutil.copy2(exp_path, TARGET_ONNX_PATH)
                    print(f"✅ Exported ONNX → {TARGET_ONNX_PATH}")
            except Exception as e:
                print(f"⚠️ ONNX export failed: {e}")
    else:
        print(f"\n⚠️ Checkpoint not found in {RUNS_DIR}. Check manually.")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF-DETR-Nano for Model 3A (Character Box Detection)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--device", type=str, default=None, help="Compute device: mps, cuda, or cpu")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX upon training completion")
    args = parser.parse_args()

    train_rfdetr_nano_charbox(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        export_onnx=args.export_onnx,
    )
