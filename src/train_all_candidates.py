"""
src/train_all_candidates.py

Unified training pipeline for Model 1 Thai License Plate Detectors:
  1. LibreDFINE-Nano    (Fastest CPU detector ~26 ms)
  2. LibreDFINE-Small   (High-accuracy compact detector ~67 ms)
  3. LibreRTDETRv2-r18  (Real-Time DETR v2)
  4. LibreRFDETR-OBB    (Oriented Bounding Box detector)

Usage:
  python src/train_all_candidates.py --all
  python src/train_all_candidates.py --model dfine_nano --epochs 25
"""

import argparse
import shutil
import time
from pathlib import Path
from libreyolo import LibreYOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BBOX_DATA_YAML = PROJECT_ROOT / "datasets" / "Thai" / "LPR_2_BBox" / "data.yaml"
OBB_DATA_YAML = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new" / "data.yaml"
OUTPUT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"


def ensure_datasets():
    if not BBOX_DATA_YAML.exists():
        from prepare_bbox_dataset import convert_polygon_to_bbox
        convert_polygon_to_bbox()


def train_dfine(size="n", epochs=20, batch=8, imgsz=640, patience=12):
    model_name = f"dfine_{'nano' if size == 'n' else 'small'}"
    pt_source = f"weights/LibreDFINE{size}.pt"
    target_pt = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.pt"
    target_onnx = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.onnx"
    save_run_dir = RUNS_DIR / f"train_{model_name}"

    print("\n" + "=" * 70)
    print(f"🚀 Training {model_name.upper()} (License: MIT)")
    print(f"   Pretrained: {pt_source}")
    print(f"   Dataset:    {BBOX_DATA_YAML}")
    print(f"   Epochs:     {epochs} | Batch: {batch} | Patience: {patience}")
    print("=" * 70)

    model = LibreYOLO(pt_source)
    model.train(
        data=str(BBOX_DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=0.0005,
        patience=patience,
        device="mps",
        project=str(save_run_dir.parent),
        name=save_run_dir.name,
        exist_ok=True,
    )

    # Copy best weights
    best_pt = save_run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(save_run_dir.glob("**/*.pt"))
        if candidates:
            best_pt = candidates[0]

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        print(f"✅ Saved fine-tuned weights → {target_pt}")
        # Export to ONNX
        trained_model = LibreYOLO(str(target_pt), device="cpu")
        trained_model.export(format="onnx", imgsz=imgsz, device="cpu", dynamic=False)
        if hasattr(trained_model, "model_path"):
            exp_onnx = Path(trained_model.model_path).with_suffix(".onnx")
            if exp_onnx.exists() and exp_onnx.resolve() != target_onnx.resolve():
                shutil.copy2(exp_onnx, target_onnx)
                print(f"✅ Exported ONNX → {target_onnx}")
    else:
        # Fallback export
        model.export(format="onnx", imgsz=imgsz, device="cpu", dynamic=False)
        print(f"✅ Exported base ONNX to weights/")


def train_rtdetrv2(epochs=20, batch=8, imgsz=640, patience=12):
    model_name = "rtdetrv2_r18"
    pt_source = "weights/LibreRTDETRv2r18.pt"
    target_pt = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.pt"
    target_onnx = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.onnx"
    save_run_dir = RUNS_DIR / f"train_{model_name}"

    print("\n" + "=" * 70)
    print(f"🚀 Training {model_name.upper()} (License: Apache-2.0)")
    print(f"   Pretrained: {pt_source}")
    print(f"   Dataset:    {BBOX_DATA_YAML}")
    print(f"   Epochs:     {epochs} | Batch: {batch} | Patience: {patience}")
    print("=" * 70)

    model = LibreYOLO(pt_source)
    model.train(
        data=str(BBOX_DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=0.0002,
        patience=patience,
        device="mps",
        project=str(save_run_dir.parent),
        name=save_run_dir.name,
        exist_ok=True,
    )

    best_pt = save_run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(save_run_dir.glob("**/*.pt"))
        if candidates:
            best_pt = candidates[0]

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        print(f"✅ Saved fine-tuned weights → {target_pt}")
        trained_model = LibreYOLO(str(target_pt))
        trained_model.export(format="onnx", imgsz=imgsz, dynamic=False)
        if hasattr(trained_model, "model_path"):
            exp_onnx = Path(trained_model.model_path).with_suffix(".onnx")
            if exp_onnx.exists() and exp_onnx.resolve() != target_onnx.resolve():
                shutil.copy2(exp_onnx, target_onnx)
                print(f"✅ Exported ONNX → {target_onnx}")


def train_rfdetr_obb(epochs=20, batch_size=4, patience=12):
    model_name = "rfdetr_obb_small"
    pt_source = "weights/LibreRFDETRs-obb.pt"
    target_pt = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.pt"
    target_onnx = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.onnx"
    save_run_dir = RUNS_DIR / f"train_{model_name}"

    print("\n" + "=" * 70)
    print(f"🚀 Training {model_name.upper()} (License: MIT)")
    print(f"   Task:       OBB (Oriented Bounding Box)")
    print(f"   Pretrained: {pt_source}")
    print(f"   Dataset:    {OBB_DATA_YAML}")
    print(f"   Epochs:     {epochs} | Batch: {batch_size} | Patience: {patience}")
    print("=" * 70)

    model = LibreYOLO(pt_source)
    model.train(
        data=str(OBB_DATA_YAML),
        epochs=epochs,
        batch=batch_size,
        lr0=1e-4,
        patience=patience,
        device="mps",
        project=str(save_run_dir.parent),
        name=save_run_dir.name,
        exist_ok=True,
    )

    best_pt = save_run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(save_run_dir.glob("**/*.pt")) + list(save_run_dir.glob("**/*.pth"))
        if candidates:
            best_pt = candidates[0]

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        print(f"✅ Saved fine-tuned weights → {target_pt}")
        trained_model = LibreYOLO(str(target_pt), device="cpu")
        trained_model.export(format="onnx", imgsz=640, device="cpu", dynamic=False)
        if hasattr(trained_model, "model_path"):
            exp_onnx = Path(trained_model.model_path).with_suffix(".onnx")
            if exp_onnx.exists() and exp_onnx.resolve() != target_onnx.resolve():
                shutil.copy2(exp_onnx, target_onnx)
                print(f"✅ Exported ONNX → {target_onnx}")


def train_picodet(size="s", epochs=30, batch=16, imgsz=416, patience=12):
    model_name = f"picodet_{size}"
    pt_source = f"weights/LibrePICODET{size}.pt"
    target_pt = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.pt"
    target_onnx = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.onnx"
    save_run_dir = RUNS_DIR / f"train_{model_name}"

    print("\n" + "=" * 70)
    print(f"🚀 Training {model_name.upper()} (PaddleDetection PicoDet / Apache-2.0)")
    print(f"   Task:       AABB (Ultra-fast Straight Rect ~15-25ms CPU)")
    print(f"   Pretrained: {pt_source}")
    print(f"   Dataset:    {BBOX_DATA_YAML}")
    print(f"   Epochs:     {epochs} | Batch: {batch} | Imgsz: {imgsz} | Patience: {patience}")
    print("=" * 70)

    model = LibreYOLO(pt_source)
    model.train(
        data=str(BBOX_DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=1e-3,
        patience=patience,
        device="mps",
        project=str(save_run_dir.parent),
        name=save_run_dir.name,
        exist_ok=True,
    )

    best_pt = save_run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(save_run_dir.glob("**/*.pt")) + list(save_run_dir.glob("**/*.pth"))
        if candidates:
            best_pt = candidates[0]

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        print(f"✅ Saved fine-tuned weights → {target_pt}")
        trained_model = LibreYOLO(str(target_pt), device="cpu")
        trained_model.export(format="onnx", imgsz=imgsz, device="cpu", dynamic=False)
        if hasattr(trained_model, "model_path"):
            exp_onnx = Path(trained_model.model_path).with_suffix(".onnx")
            if exp_onnx.exists() and exp_onnx.resolve() != target_onnx.resolve():
                shutil.copy2(exp_onnx, target_onnx)
                print(f"✅ Exported ONNX → {target_onnx}")


def train_rtdetrv2_obb(epochs=25, batch_size=4, imgsz=1024, patience=12):
    model_name = "rtdetrv2_obb_small"
    pt_source = "weights/LibreRTDETRv2s-obb.pt"
    target_pt = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.pt"
    target_onnx = OUTPUT_WEIGHTS_DIR / f"plate_detector_{model_name}.onnx"
    save_run_dir = RUNS_DIR / f"train_{model_name}"

    print("\n" + "=" * 70)
    print(f"ℹ️  RT-DETRv2 OBB is inference-only in LibreYOLO (upstream training head is not open-sourced)")
    print(f"   Pretrained checkpoint: {pt_source}")
    print(f"   Exporting pretrained ONNX for C# evaluation directly...")
    print("=" * 70)

    trained_model = LibreYOLO(pt_source, device="cpu")
    trained_model.export(format="onnx", imgsz=imgsz, device="cpu", dynamic=False)
    exp_onnx = Path("weights/LibreRTDETRv2s-obb.onnx")
    if exp_onnx.exists() and exp_onnx.resolve() != target_onnx.resolve():
        shutil.copy2(exp_onnx, target_onnx)
        print(f"✅ Exported ONNX → {target_onnx}")
    return

    best_pt = save_run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        candidates = list(save_run_dir.glob("**/*.pt")) + list(save_run_dir.glob("**/*.pth"))
        if candidates:
            best_pt = candidates[0]

    if best_pt and best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        print(f"✅ Saved fine-tuned weights → {target_pt}")
        trained_model = LibreYOLO(str(target_pt), device="cpu")
        trained_model.export(format="onnx", imgsz=imgsz, device="cpu", dynamic=False)
        if hasattr(trained_model, "model_path"):
            exp_onnx = Path(trained_model.model_path).with_suffix(".onnx")
            if exp_onnx.exists() and exp_onnx.resolve() != target_onnx.resolve():
                shutil.copy2(exp_onnx, target_onnx)
                print(f"✅ Exported ONNX → {target_onnx}")


def main():
    parser = argparse.ArgumentParser(description="Train Model 1 Plate Detector Candidates")
    parser.add_argument("--model", type=str, default="all",
                        help="Model(s) to train: dfine_nano, dfine_small, rtdetrv2, rfdetr_obb, picodet_s, rtdetrv2_obb, or comma-separated list e.g. 'rfdetr_obb,picodet_s,rtdetrv2_obb' or 'all'")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs (default: 45)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--patience", type=int, default=12,
                        help="Early stopping patience: stop if no validation improvement for N epochs (default: 12)")
    args = parser.parse_args()

    ensure_datasets()
    OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    selected_models = [m.strip() for m in args.model.split(",") if m.strip()]
    run_all = "all" in selected_models

    t_start = time.time()

    if run_all or "dfine_nano" in selected_models:
        train_dfine(size="n", epochs=args.epochs, batch=args.batch, patience=args.patience)

    if run_all or "dfine_small" in selected_models:
        train_dfine(size="s", epochs=args.epochs, batch=args.batch, patience=args.patience)

    if run_all or "rtdetrv2" in selected_models:
        train_rtdetrv2(epochs=args.epochs, batch=args.batch, patience=args.patience)

    if run_all or "rfdetr_obb" in selected_models:
        train_rfdetr_obb(epochs=args.epochs, batch_size=max(2, args.batch // 2), patience=args.patience)

    if run_all or "picodet_s" in selected_models:
        train_picodet(size="s", epochs=args.epochs, batch=max(8, args.batch * 2), patience=args.patience)

    if run_all or "rtdetrv2_obb" in selected_models:
        train_rtdetrv2_obb(epochs=args.epochs, batch_size=max(2, args.batch // 2), patience=args.patience)

    total_mins = (time.time() - t_start) / 60
    print("\n" + "=" * 70)
    print(f"🎉 All requested training tasks finished in {total_mins:.1f} minutes!")
    print("=" * 70)


if __name__ == "__main__":
    main()
