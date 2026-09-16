"""
src/train_librerfdetr_obb_plate.py

Fine-tunes LibreRFDETR-Small OBB (MIT License) for Model 1 Oriented Plate Detection.
Outputs:
  - PyTorch: weights/plate_detector_rfdetr_obb_small.pt
  - ONNX:    weights/plate_detector_rfdetr_obb_small.onnx
"""

import shutil
from pathlib import Path
from libreyolo import LibreYOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new"
DATASET_YAML = DATASET_DIR / "data.yaml"
OUTPUT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
TARGET_PT = OUTPUT_WEIGHTS_DIR / "plate_detector_rfdetr_obb_small.pt"
TARGET_ONNX = OUTPUT_WEIGHTS_DIR / "plate_detector_rfdetr_obb_small.onnx"
RUNS_DIR = PROJECT_ROOT / "runs" / "train_rfdetr_obb" / "plate"


def train_rfdetr_obb_plate(epochs=20, batch_size=4):
    print("=" * 70)
    print("🚀 LibreRFDETR-Small OBB Training — Model 1 (Oriented Plate Detection)")
    print("   License    : MIT ✅ (100% free commercial use)")
    print(f"   Task       : OBB (Oriented Bounding Box — direct plate un-rotation)")
    print(f"   Config     : {DATASET_YAML}")
    print(f"   Target PT  : {TARGET_PT}")
    print(f"   Target ONNX: {TARGET_ONNX}")
    print("=" * 70)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load pretrained LibreRFDETRs-obb
    model = LibreYOLO("weights/LibreRFDETRs-obb.pt")

    # 2. Train on polygon/obb dataset
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4,
        output_dir=str(RUNS_DIR),
        accelerator="mps",
    )

    # 3. Locate best checkpoint
    candidates = list(RUNS_DIR.glob("**/*.pth")) + list(RUNS_DIR.glob("**/*.pt"))
    best_pt = candidates[0] if candidates else None

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, TARGET_PT)
        print(f"✅ Saved fine-tuned weights → {TARGET_PT}")

        # 4. Export to ONNX for C# integration
        trained_model = LibreYOLO(str(TARGET_PT))
        trained_model.export(format="onnx", imgsz=640, dynamic=False)
        exported = trained_model.model_path.with_suffix(".onnx") if hasattr(trained_model, "model_path") else None
        if exported and exported.exists():
            shutil.copy2(exported, TARGET_ONNX)
            print(f"✅ Saved optimized ONNX → {TARGET_ONNX}")
    else:
        print("⚠️ Checkpoint search fallback, export baseline weights to ONNX...")
        model.export(format="onnx", imgsz=640, dynamic=False)

    print("=" * 70)


if __name__ == "__main__":
    train_rfdetr_obb_plate()
