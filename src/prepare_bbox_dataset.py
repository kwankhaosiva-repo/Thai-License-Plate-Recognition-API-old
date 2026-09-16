"""
src/prepare_bbox_dataset.py

Converts 4-point polygon labels in 'datasets/Thai/LPR 2 - Polygon.yolov11_new'
into standard YOLO bounding boxes (class x_center y_center width height)
for LibreDFINE training, saving to 'datasets/Thai/LPR_2_BBox/'.
"""

import os
import shutil
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DATASET_DIR = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new"
DST_DATASET_DIR = PROJECT_ROOT / "datasets" / "Thai" / "LPR_2_BBox"


def convert_polygon_to_bbox():
    print("=" * 70)
    print("📦 Converting Polygon Dataset to Standard BBox for D-FINE")
    print(f"   Source : {SRC_DATASET_DIR}")
    print(f"   Target : {DST_DATASET_DIR}")
    print("=" * 70)

    if not SRC_DATASET_DIR.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_DATASET_DIR}")

    DST_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]
    total_converted = 0

    for split in splits:
        src_img_dir = SRC_DATASET_DIR / split / "images"
        src_lbl_dir = SRC_DATASET_DIR / split / "labels"

        dst_img_dir = DST_DATASET_DIR / split / "images"
        dst_lbl_dir = DST_DATASET_DIR / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_lbl_dir.exists():
            print(f"Skipping {split} (labels dir not found)")
            continue

        label_files = list(src_lbl_dir.glob("*.txt"))
        print(f"Processing {split}: {len(label_files)} label files...")

        for lbl_path in label_files:
            img_stem = lbl_path.stem
            matched_img = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                candidate = src_img_dir / f"{img_stem}{ext}"
                if candidate.exists():
                    matched_img = candidate
                    break

            if matched_img:
                dst_img = dst_img_dir / matched_img.name
                if not dst_img.exists():
                    try:
                        os.symlink(matched_img.resolve(), dst_img)
                    except (OSError, AttributeError):
                        shutil.copy2(matched_img, dst_img)

            converted_lines = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = parts[0]
                    coords = [float(v) for v in parts[1:]]

                    if len(coords) == 4:
                        converted_lines.append(f"{cls_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    elif len(coords) >= 6 and len(coords) % 2 == 0:
                        xs = coords[0::2]
                        ys = coords[1::2]
                        min_x, max_x = max(0.0, min(xs)), min(1.0, max(xs))
                        min_y, max_y = max(0.0, min(ys)), min(1.0, max(ys))
                        w = max(1e-4, max_x - min_x)
                        h = max(1e-4, max_y - min_y)
                        xc = min_x + w / 2.0
                        yc = min_y + h / 2.0
                        converted_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            dst_lbl_file = dst_lbl_dir / lbl_path.name
            with open(dst_lbl_file, "w", encoding="utf-8") as f:
                f.writelines(converted_lines)
            total_converted += 1

    data_yaml = {
        "train": str((DST_DATASET_DIR / "train" / "images").resolve()),
        "val": str((DST_DATASET_DIR / "valid" / "images").resolve()),
        "test": str((DST_DATASET_DIR / "test" / "images").resolve()),
        "nc": 1,
        "names": ["plate"],
    }
    with open(DST_DATASET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"✅ Conversion complete! Converted {total_converted} files.")
    print(f"   Config saved to {DST_DATASET_DIR / 'data.yaml'}")
    print("=" * 70)


if __name__ == "__main__":
    convert_polygon_to_bbox()
