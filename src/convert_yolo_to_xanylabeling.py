"""
src/convert_yolo_to_xanylabeling.py

Convert YOLO annotations (.txt) to X-AnyLabeling / LabelMe compliant (.json) format.
Supports:
  1. Standard YOLO Bounding Box (cls xc yc w h) -> rectangle or 4-point polygon
  2. YOLO Polygon / OBB (cls x1 y1 x2 y2 x3 y3 x4 y4) -> polygon
  3. Automatic class name resolution via data.yaml or classes.txt
  4. Writes .json directly beside images so X-AnyLabeling loads annotations instantly.

Usage:
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/convert_yolo_to_xanylabeling.py \
      --dataset-dir "datasets/Thai/LPR_2_BBox"

  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/convert_yolo_to_xanylabeling.py \
      --dataset-dir "datasets/Thai/LPR 2 - Polygon.yolov11_new"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image


def load_class_names(dataset_dir: Path) -> Dict[int, str]:
    """Resolves class id to class name mapping from data.yaml or classes.txt."""
    search_dirs = [dataset_dir, dataset_dir.parent, dataset_dir.parent.parent]
    for d in search_dirs:
        data_yaml = d / "data.yaml"
        if data_yaml.exists():
            try:
                with open(data_yaml, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    names = data.get("names", [])
                    if isinstance(names, list):
                        return {i: str(name) for i, name in enumerate(names)}
                    elif isinstance(names, dict):
                        return {int(k): str(v) for k, v in names.items()}
            except Exception as e:
                print(f"Warning: Failed to parse data.yaml in {d} ({e})")

        classes_txt = d / "classes.txt"
        if classes_txt.exists():
            try:
                with open(classes_txt, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    return {i: line for i, line in enumerate(lines)}
            except Exception as e:
                print(f"Warning: Failed to parse classes.txt in {d} ({e})")

    return {}


def yolo_box_to_points(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> List[List[float]]:
    """Converts normalized YOLO xc, yc, w, h into 4-point rectangle."""
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return [
        [round(float(x1), 2), round(float(y1), 2)],
        [round(float(x2), 2), round(float(y1), 2)],
        [round(float(x2), 2), round(float(y2), 2)],
        [round(float(x1), 2), round(float(y2), 2)],
    ]


def yolo_poly_to_points(coords: List[float], img_w: int, img_h: int) -> List[List[float]]:
    """Converts normalized polygon points [x1, y1, x2, y2, ...] to pixel coordinates."""
    points = []
    for i in range(0, len(coords), 2):
        px = coords[i] * img_w
        py = coords[i + 1] * img_h
        points.append([round(float(px), 2), round(float(py), 2)])
    return points


def convert_dataset_split(
    split_dir: Path,
    class_map: Dict[int, str],
    place_beside_image: bool = True,
    overwrite: bool = True,
) -> Tuple[int, int]:
    """
    Finds image files, pairs them with their YOLO label .txt, and writes X-AnyLabeling JSONs.
    Returns (num_converted, num_skipped).
    """
    img_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
    img_files = []
    for ext in img_extensions:
        img_files.extend(split_dir.rglob(ext))

    num_converted = 0
    num_skipped = 0

    for img_path in img_files:
        # Determine candidate label paths
        # Structure 1: images/foo.jpg and labels/foo.txt
        # Structure 2: foo.jpg and foo.txt in the same directory
        stem = img_path.stem
        candidate_label_1 = img_path.parent.parent / "labels" / f"{stem}.txt"
        candidate_label_2 = img_path.parent / f"{stem}.txt"

        label_path = None
        if candidate_label_1.exists():
            label_path = candidate_label_1
        elif candidate_label_2.exists():
            label_path = candidate_label_2

        # Output JSON path: by default right next to the image
        json_out_path = img_path.parent / f"{stem}.json"
        if json_out_path.exists() and not overwrite:
            num_skipped += 1
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"Skipping corrupt image {img_path}: {e}")
            num_skipped += 1
            continue

        shapes = []
        if label_path and label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        label_name = class_map.get(cls_id, str(cls_id))

                        if len(parts) == 5:
                            # Standard Bounding Box
                            xc, yc, w, h = map(float, parts[1:5])
                            pts = yolo_box_to_points(xc, yc, w, h, img_w, img_h)
                            shapes.append(
                                {
                                    "label": label_name,
                                    "score": 1.0,
                                    "points": pts,
                                    "group_id": None,
                                    "description": "",
                                    "difficult": False,
                                    "shape_type": "rectangle",
                                    "flags": {},
                                    "attributes": {},
                                    "kie_linking": [],
                                }
                            )
                        else:
                            # Polygon (e.g. 8 values for 4 corners)
                            coords = list(map(float, parts[1:]))
                            pts = yolo_poly_to_points(coords, img_w, img_h)
                            shapes.append(
                                {
                                    "label": label_name,
                                    "score": 1.0,
                                    "points": pts,
                                    "group_id": None,
                                    "description": "",
                                    "difficult": False,
                                    "shape_type": "polygon",
                                    "flags": {},
                                    "attributes": {},
                                    "kie_linking": [],
                                }
                            )
                    except Exception as parse_err:
                        print(f"Error parsing line in {label_path}: {parse_err}")

        # Construct X-AnyLabeling JSON
        data = {
            "version": "4.2.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": img_h,
            "imageWidth": img_w,
        }

        with open(json_out_path, "w", encoding="utf-8") as jf:
            json.dump(data, jf, ensure_ascii=False, indent=2)

        num_converted += 1

    return num_converted, num_skipped


def clean_json_annotations(dataset_dir: Path) -> int:
    """Removes all generated .json files in dataset_dir."""
    count = 0
    for jf in dataset_dir.rglob("*.json"):
        if not jf.name.endswith(".coco.json"):
            jf.unlink()
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO annotations (.txt) to X-AnyLabeling compliant (.json) format."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to YOLO dataset directory (e.g. datasets/Thai/LPR_2_BBox)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean/delete generated .json files instead of converting",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing .json files",
    )
    args = parser.parse_args()

    ds_path = Path(args.dataset_dir).resolve()
    if not ds_path.exists():
        print(f"Error: Dataset directory does not exist: {ds_path}")
        return

    if args.clean:
        cleaned = clean_json_annotations(ds_path)
        print(f"Removed {cleaned} .json files from {ds_path}")
        return

    class_map = load_class_names(ds_path)
    print(f"Target dataset: {ds_path}")
    print(f"Resolved classes: {class_map}")

    conv, skip = convert_dataset_split(
        split_dir=ds_path,
        class_map=class_map,
        overwrite=not args.no_overwrite,
    )
    print(f"Finished: {conv} images converted to X-AnyLabeling JSONs ({skip} skipped).")


if __name__ == "__main__":
    main()
