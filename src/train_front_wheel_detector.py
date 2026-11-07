"""
src/train_front_wheel_detector.py

Trains a 2-class Front Cab Face & Steer Wheel Detector (v6):
  - Class 0: cab_face (Front bumper & cab face without side mirrors or trailers)
  - Class 1: front_wheel (Ground-contact front steer tires)

Architecture: YOLO11n (Permissive License, ultra-fast on edge CPU/MPS)
Resolution: 640x640 (aspect-ratio-preserving letterbox)
"""

import sys
import shutil
import json
import random
from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_YAML = PROJECT_ROOT / "datasets" / "vehicle_front_wheel_v6" / "data.yaml"
OUTPUT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs" / "train_front_wheel_v6"


def prepare_dataset(split_ratio: float = 0.80, seed: int = 42):
    raw_dir = PROJECT_ROOT / "chacrit" / "raw_car" / "selected_image"
    dataset_dir = PROJECT_ROOT / "datasets" / "vehicle_front_wheel_v6"

    img_train_dir = dataset_dir / "images" / "train"
    img_val_dir = dataset_dir / "images" / "val"
    lbl_train_dir = dataset_dir / "labels" / "train"
    lbl_val_dir = dataset_dir / "labels" / "val"

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if f.is_file():
                f.unlink()

    json_files = sorted(list(raw_dir.glob("*.json")))
    random.seed(seed)
    shuffled = json_files.copy()
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * split_ratio)
    train_files = set(shuffled[:n_train])

    class_map = {"cab_face": 0, "front_wheel": 1}
    stats = {"train_cab": 0, "train_wheel": 0, "val_cab": 0, "val_wheel": 0}

    for jf in json_files:
        is_train = jf in train_files
        split_name = "train" if is_train else "val"

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w = data.get("imageWidth", 1280)
        img_h = data.get("imageHeight", 720)

        img_name = jf.stem + ".jpg"
        src_img = raw_dir / img_name
        if not src_img.exists():
            continue

        dst_img = (img_train_dir if is_train else img_val_dir) / img_name
        shutil.copy2(src_img, dst_img)

        lines = []
        for s in data.get("shapes", []):
            label = s.get("label")
            if label not in class_map:
                continue
            cls_id = class_map[label]
            pts = s.get("points", [])
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if label == "cab_face":
                stats[f"{split_name}_cab"] += 1
            else:
                stats[f"{split_name}_wheel"] += 1

        dst_lbl = (lbl_train_dir if is_train else lbl_val_dir) / f"{jf.stem}.txt"
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))

    print(f"📦 Dataset Partitioning Complete: {len(train_files)} Train, {len(json_files) - len(train_files)} Val")
    print(f"   Train: {stats['train_cab']} cab_face, {stats['train_wheel']} front_wheel")
    print(f"   Val:   {stats['val_cab']} cab_face, {stats['val_wheel']} front_wheel")


def train_detector(
    epochs: int = 70,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "mps",
):
    print("=" * 70)
    print("🚀 Training Front Cab Face & Steer Wheel Detector (v6)")
    print(f"   Architecture: YOLO11n | Imgsz: {imgsz}x{imgsz} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"   Device: {device}")
    print("=" * 70)

    # 0. Convert updated annotations & partition dataset
    prepare_dataset()

    # 1. Load pretrained YOLO11n
    base_model = PROJECT_ROOT / "yolo11n.pt"
    if not base_model.exists():
        base_model = "yolo11n.pt"
    model = YOLO(str(base_model))

    # 2. Train with physics-consistent augmentations
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(RUNS_DIR.parent),
        name=RUNS_DIR.name,
        exist_ok=True,
        # Aspect-ratio & Geometric augmentations
        rect=False,             # Letterbox training
        fliplr=0.5,             # Symmetric horizontal flipping
        flipud=0.0,             # No vertical flip
        degrees=10.0,           # Camera perspective tilt
        translate=0.10,         # Frame shift
        scale=0.50,             # Multi-scale distance (near vs far)
        shear=2.0,              # Camera skew
        perspective=0.0005,     # Perspective road projection
        # Photometric augmentations (day / night / shadows)
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        mosaic=0.50,            # Context mixing
        mixup=0.0,
        # Optimization
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        warmup_epochs=3.0,
        save=True,
        plots=True,
        verbose=True,
        seed=42,
    )

    best_pt = RUNS_DIR / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = RUNS_DIR / "weights" / "last.pt"

    print("\n" + "=" * 70)
    print(f"✅ Training completed! Best checkpoint: {best_pt}")
    print("=" * 70)

    # 3. Export to weights/ directory
    OUTPUT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dst_pt = OUTPUT_WEIGHTS_DIR / "front_wheel_detector_v6.pt"
    shutil.copy2(best_pt, dst_pt)
    print(f"📦 Saved PyTorch weights to: {dst_pt}")

    # 4. Evaluate on Validation Set
    print("\n📊 Running final validation evaluation on held-out test set (51 images)...")
    eval_model = YOLO(str(dst_pt))
    metrics = eval_model.val(data=str(DATA_YAML), imgsz=imgsz, device=device, split="val")

    print("\n" + "=" * 70)
    print("🏆 FINAL VALIDATION METRICS:")
    print(f"   mAP50 (All):       {metrics.box.map50 * 100:.2f}%")
    print(f"   mAP50-95 (All):    {metrics.box.map * 100:.2f}%")
    print(f"   Precision:         {metrics.box.mp * 100:.2f}%")
    print(f"   Recall:            {metrics.box.mr * 100:.2f}%")

    for i, name in enumerate(["cab_face", "front_wheel"]):
        if i < len(metrics.box.maps):
            map50 = metrics.box.map50_per_class[i] if hasattr(metrics.box, "map50_per_class") else metrics.box.maps[i]
            print(f"   Class '{name:<12}': mAP50 = {map50 * 100:.2f}%")
    print("=" * 70)

    # 5. Export standalone ONNX (Opset 18)
    print("\n📦 Exporting standalone ONNX model (Opset 18)...")
    onnx_path = eval_model.export(format="onnx", opset=18, dynamic=True, imgsz=imgsz)
    dst_onnx = OUTPUT_WEIGHTS_DIR / "front_wheel_detector_v6.onnx"
    if Path(onnx_path).resolve() != dst_onnx.resolve():
        shutil.copy2(onnx_path, dst_onnx)
    print(f"✅ Saved ONNX model to: {dst_onnx} ({dst_onnx.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_detector(epochs=60, batch_size=16, imgsz=640, device=device)
