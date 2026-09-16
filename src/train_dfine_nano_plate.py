"""
src/train_dfine_nano_plate.py

Fine-tunes D-FINE Nano (MIT License) for Model 1 Thai Plate Detection.
Classes: plate (1 class).
Saves weights to weights/plate_detector_dfine_nano.pt and .onnx.
"""

import argparse
import shutil
from pathlib import Path
from libreyolo import LibreYOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BBOX_DATA_YAML      = PROJECT_ROOT / "datasets" / "Thai" / "LPR_2_BBox" / "data.yaml"
OUTPUT_WEIGHTS_DIR  = PROJECT_ROOT / "weights"
TARGET_WEIGHTS_PATH = OUTPUT_WEIGHTS_DIR / "plate_detector_dfine_nano.pt"
TARGET_ONNX_PATH    = OUTPUT_WEIGHTS_DIR / "plate_detector_dfine_nano.onnx"
RUNS_DIR            = PROJECT_ROOT / "runs" / "train_dfine_nano" / "plate"
PRETRAINED_SOURCE   = OUTPUT_WEIGHTS_DIR / "LibreDFINEn.pt"


def ensure_bbox_dataset():
    if not BBOX_DATA_YAML.exists():
        from prepare_bbox_dataset import convert_polygon_to_bbox
        convert_polygon_to_bbox()


def train_dfine_nano_plate(epochs=30, batch=8, imgsz=640, patience=12, device="mps", export_onnx=True):
    print("=" * 70)
    print("🚀 D-FINE Nano Training — Model 1 (Plate Detection)")
    print(f"   License   : MIT ✅ (Commercially permissive)")
    print(f"   Classes   : plate (1 class)")
    print(f"   Pretrained: {PRETRAINED_SOURCE}")
    print(f"   Dataset   : {BBOX_DATA_YAML}")
    print(f"   Output    : {TARGET_WEIGHTS_PATH}")
    print(f"   Epochs    : {epochs} | Batch: {batch} | Imgsz: {imgsz} | Patience: {patience}")
    print("=" * 70)

    ensure_bbox_dataset()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = LibreYOLO(str(PRETRAINED_SOURCE))
    model.train(
        data=str(BBOX_DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=0.0005,
        patience=patience,
        device=device,
        project=str(RUNS_DIR.parent),
        name=RUNS_DIR.name,
        exist_ok=True,
    )

    best_pt = RUNS_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(RUNS_DIR.glob("**/*.pt")) + list(RUNS_DIR.glob("**/*.pth"))
        if candidates:
            best_pt = candidates[0]

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, TARGET_WEIGHTS_PATH)
        size_mb = TARGET_WEIGHTS_PATH.stat().st_size / 1e6
        print(f"\n✅ Saved best fine-tuned weights → {TARGET_WEIGHTS_PATH} ({size_mb:.1f} MB)")

        if export_onnx:
            print(f"Exporting to standalone ONNX (CPU isolated) → {TARGET_ONNX_PATH}...")
            try:
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

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D-FINE Nano for Model 1 (Plate Detection)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    parser.add_argument("--device", type=str, default="mps", help="Device: mps, cuda, or cpu (default: mps)")
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false", help="Skip ONNX export")
    args = parser.parse_args()

    train_dfine_nano_plate(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        export_onnx=args.export_onnx,
    )
