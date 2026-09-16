"""
src/train_libredfine_plate.py

Fine-tunes LibreDFINE-Nano (MIT License) for Model 1 Thai Plate Detection.
Outputs:
  - PyTorch: weights/plate_detector_dfine_nano.pt
  - ONNX:    weights/plate_detector_dfine_nano.onnx
"""

import shutil
from pathlib import Path
from libreyolo import LibreYOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "datasets" / "Thai" / "LPR_2_BBox" / "data.yaml"
OUTPUT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
TARGET_PT = OUTPUT_WEIGHTS_DIR / "plate_detector_dfine_nano.pt"
TARGET_ONNX = OUTPUT_WEIGHTS_DIR / "plate_detector_dfine_nano.onnx"
RUNS_DIR = PROJECT_ROOT / "runs" / "train_dfine" / "plate"


def train_dfine_plate(epochs=20, batch=8, imgsz=640):
    print("=" * 70)
    print("🚀 LibreDFINE-Nano Training — Model 1 (Plate Detection)")
    print("   License   : MIT ✅ (100% free commercial use)")
    print(f"   Config    : {DATASET_YAML}")
    print(f"   Target PT : {TARGET_PT}")
    print(f"   Target ONNX: {TARGET_ONNX}")
    print("=" * 70)

    if not DATASET_YAML.exists():
        # Automatically generate bbox dataset if missing
        from prepare_bbox_dataset import convert_polygon_to_bbox
        convert_polygon_to_bbox()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load pretrained LibreDFINEn
    model = LibreYOLO("weights/LibreDFINEn.pt")

    # 2. Train on converted bbox dataset
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=0.0005,
        device="mps",
        project=str(RUNS_DIR.parent),
        name=RUNS_DIR.name,
        exist_ok=True,
    )

    # 3. Locate best checkpoint
    best_pt = RUNS_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(RUNS_DIR.glob("**/*.pt"))
        if candidates:
            best_pt = candidates[0]

    if best_pt.exists():
        shutil.copy2(best_pt, TARGET_PT)
        print(f"✅ Saved fine-tuned weights → {TARGET_PT}")

        # 4. Export to ONNX for C# integration
        trained_model = LibreYOLO(str(TARGET_PT))
        trained_model.export(format="onnx", imgsz=imgsz, dynamic=False)
        exported = trained_model.model_path.with_suffix(".onnx") if hasattr(trained_model, "model_path") else None
        if exported and exported.exists():
            shutil.copy2(exported, TARGET_ONNX)
            print(f"✅ Saved optimized ONNX → {TARGET_ONNX}")
    else:
        print("⚠️ Training completed, export baseline weights to ONNX...")
        model.export(format="onnx", imgsz=imgsz, dynamic=False)

    print("=" * 70)


if __name__ == "__main__":
    train_dfine_plate()
