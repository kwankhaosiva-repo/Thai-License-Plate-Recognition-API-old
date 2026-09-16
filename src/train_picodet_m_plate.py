"""
src/train_picodet_m_plate.py

Fine-tunes PaddleDetection PicoDet-M (Apache-2.0 License) for Model 1 Car Plate Detection.
Saves weights to:
  - PyTorch: weights/plate_detector_picodet_m.pt
  - ONNX:    weights/plate_detector_picodet_m.onnx
"""

import argparse
import shutil
import time
from pathlib import Path
import torch

from libreyolo import LibreYOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BBOX_DATA_YAML      = PROJECT_ROOT / "datasets" / "Thai" / "LPR_2_BBox" / "data.yaml"
OUTPUT_WEIGHTS_DIR  = PROJECT_ROOT / "weights"
TARGET_WEIGHTS_PATH = OUTPUT_WEIGHTS_DIR / "plate_detector_picodet_m.pt"
TARGET_ONNX_PATH    = OUTPUT_WEIGHTS_DIR / "plate_detector_picodet_m.onnx"
RUNS_DIR            = PROJECT_ROOT / "runs" / "train_picodet_m"
PRETRAINED_SOURCE   = OUTPUT_WEIGHTS_DIR / "LibrePICODETm.pt"


def ensure_dataset():
    """Ensure standard bbox dataset exists (converts from polygon if needed)."""
    if not BBOX_DATA_YAML.exists():
        print("BBox dataset not found, generating from polygon dataset...")
        from prepare_bbox_dataset import convert_polygon_to_bbox
        convert_polygon_to_bbox()


def ensure_pretrained_weights():
    """Ensure LibrePICODETm.pt is available in weights/."""
    if not PRETRAINED_SOURCE.exists():
        print(f"Downloading pretrained PicoDet-M checkpoint to {PRETRAINED_SOURCE}...")
        import urllib.request
        url = "https://huggingface.co/LibreYOLO/LibrePICODETm/resolve/main/LibrePICODETm.pt"
        urllib.request.urlretrieve(url, str(PRETRAINED_SOURCE))
        print("Download complete!")


def train_picodet_m_plate(
    epochs: int = 35,
    batch: int = 16,
    imgsz: int = 416,
    lr: float = 1e-3,
    patience: int = 12,
    device: str = "mps",
    export_onnx: bool = True,
):
    print("=" * 75)
    print("🚀 PicoDet-M Training — Model 1 (Thai License Plate Detection)")
    print(f"   Architecture : PaddleDetection PicoDet-M (ESNet-M + CSPPAN, stacked_convs=4)")
    print(f"   License      : Apache-2.0 ✅ (100% Commercial Permissive)")
    print(f"   Pretrained   : {PRETRAINED_SOURCE.name}")
    print(f"   Dataset      : {BBOX_DATA_YAML}")
    print(f"   Resolution   : {imgsz}x{imgsz}")
    print(f"   Epochs       : {epochs} | Batch: {batch} | Patience: {patience}")
    print(f"   Device       : {device.upper()}")
    print("=" * 75)

    ensure_dataset()
    ensure_pretrained_weights()

    OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Initialize model from pretrained weights
    model = LibreYOLO(str(PRETRAINED_SOURCE))

    # Train on Thai plate dataset
    model.train(
        data=str(BBOX_DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr,
        patience=patience,
        device=device,
        project=str(RUNS_DIR.parent),
        name=RUNS_DIR.name,
        exist_ok=True,
    )

    # Locate best trained checkpoint
    best_pt = RUNS_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(RUNS_DIR.glob("**/*.pt")) + list(RUNS_DIR.glob("**/*.pth"))
        if candidates:
            best_pt = candidates[0]

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, TARGET_WEIGHTS_PATH)
        size_mb = TARGET_WEIGHTS_PATH.stat().st_size / 1e6
        print(f"\n✅ Saved fine-tuned weights → {TARGET_WEIGHTS_PATH} ({size_mb:.1f} MB)")

        if export_onnx:
            print(f"Exporting to standalone ONNX (CPU isolated) → {TARGET_ONNX_PATH}...")
            try:
                # Force export on CPU to avoid MPS constant-folding JIT issues
                trained_model = LibreYOLO(str(TARGET_WEIGHTS_PATH), device="cpu")
                trained_model.export(format="onnx", imgsz=imgsz, device="cpu", dynamic=False)

                if hasattr(trained_model, "model_path"):
                    exp_onnx = Path(trained_model.model_path).with_suffix(".onnx")
                    if exp_onnx.exists() and exp_onnx.resolve() != TARGET_ONNX_PATH.resolve():
                        shutil.copy2(exp_onnx, TARGET_ONNX_PATH)

                if TARGET_ONNX_PATH.exists():
                    onnx_mb = TARGET_ONNX_PATH.stat().st_size / 1e6
                    print(f"✅ Exported ONNX → {TARGET_ONNX_PATH} ({onnx_mb:.1f} MB)")
            except Exception as e:
                print(f"⚠️ ONNX export note: {e}")
    else:
        print(f"⚠️ Checkpoint not found in {RUNS_DIR}. Check logs.")

    elapsed_mins = (time.time() - t_start) / 60
    print(f"\n🎉 PicoDet-M Training completed in {elapsed_mins:.1f} minutes!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PicoDet-M for Model 1 Plate Detection")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs (default: 35)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image resolution (default: 416)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    parser.add_argument("--device", type=str, default="mps", help="Compute device: mps, cuda, or cpu (default: mps)")
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false", help="Skip automatic ONNX export")
    args = parser.parse_args()

    train_picodet_m_plate(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        export_onnx=args.export_onnx,
    )
