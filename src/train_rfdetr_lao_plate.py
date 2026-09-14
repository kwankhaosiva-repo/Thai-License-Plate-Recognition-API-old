"""
src/train_rfdetr_lao_plate.py

Fine-tunes RF-DETR-Small (Apache-2.0) for Lao Plate Detection.
Same dataset as original RT-DETR enterprise plan: datasets/Lao/laos plate.v3i.yolov11/
Output: weights/plate_detector_lao_rfdetr.pt
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR         = PROJECT_ROOT / "datasets" / "Lao" / "laos plate.v3i.yolov11"
DATASET_YAML        = DATASET_DIR / "data.yaml"
OUTPUT_WEIGHTS_DIR  = PROJECT_ROOT / "weights"
TARGET_WEIGHTS_PATH = OUTPUT_WEIGHTS_DIR / "plate_detector_lao_rfdetr.pt"
RUNS_DIR            = PROJECT_ROOT / "runs" / "train_rfdetr" / "lao_plate"


def train_rfdetr_lao_plate():
    print("=" * 70)
    print("🚀 RF-DETR-Base Training — Lao Plate Detection")
    print(f"   License   : Apache-2.0 ✅")
    print(f"   Classes   : plate (1 class)")
    print(f"   Dataset   : {DATASET_DIR.name}  (165 images)")
    print(f"   Config    : {DATASET_YAML}")
    print(f"   Output    : {TARGET_WEIGHTS_PATH}")
    print("=" * 70)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATASET_YAML}")

    from rfdetr import RFDETRBase

    print("\n[1/3] Loading RF-DETR-Base pretrained model (Apache-2.0)...")
    model = RFDETRBase()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    model.train(
        dataset_dir=str(DATASET_DIR),
        dataset_file="yolo",
        epochs=30,
        batch_size=4,
        grad_accum_steps=2,
        multi_scale=False,         # fixes 27GB RAM swap thrashing, stays at 560px
        lr=1e-4,
        lr_encoder=1e-5,
        output_dir=str(RUNS_DIR),
        use_ema=True,
        checkpoint_interval=1,
        early_stopping=True,
        early_stopping_patience=10,
        accelerator="mps",
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
        print(f"\n[3/3] ✅ Saved → {TARGET_WEIGHTS_PATH} ({size_mb:.1f} MB)")
    else:
        print(f"\n⚠️  Checkpoint not found in {RUNS_DIR}.")
    print("=" * 70)


if __name__ == "__main__":
    train_rfdetr_lao_plate()
