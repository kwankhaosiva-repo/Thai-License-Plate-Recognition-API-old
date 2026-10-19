"""
src/convert_coco_to_yolo.py

Convert a Roboflow COCO export into the YOLO layout that
`src/train_libreyolo_task.py --task charbox` expects:

    <out>/train/images/*.jpg
    <out>/train/labels/*.txt      one "cls xc yc w h" line per box, normalised
    <out>/data.yaml

WHY
===
`datasets/Thai/LPR 2 - Character Box Detection.coco` is a COCO export:
    train/valid/test/*.jpg  +  _annotations.coco.json
with TWO categories (id 0 = the dataset itself, id 1 = "Each Thai Charactor car
plate"). The trainer and the whole LibreYOLO path are YOLO-only, and Roboflow
counts the parent class as a category — so class 0 must be DROPPED, not trained
on, or the detector learns to emit empty boxes over the whole image.

MEASURED (see docs/RETRAIN_V6.md): this export carries the SAME 657 source
images as the old `.yolov11` folder and the box geometry differs by only ~3%
(area 0.1144 -> 0.1178, aspect 0.578 -> 0.597). It is a FORMAT change, not a
geometry change — see the answer to "is the new bbox wider/squarer?".

USAGE
=====
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/convert_coco_to_yolo.py \
      --src "datasets/Thai/LPR 2 - Character Box Detection.coco" \
      --out "datasets/Thai/LPR 2 - Character Box Detection.yolov8" \
      --class-id 1 --class-name char

  ... --dry-run        report class counts / geometry, write nothing
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def convert(src: Path, out: Path, keep_class_id: int, class_name: str, dry_run: bool):
    stats = Counter()
    aspects, areas, n_img = [], [], 0

    for split in ("train", "valid", "test"):
        idir = src / split
        ann = idir / "_annotations.coco.json"
        if not ann.exists():
            print(f"  skip {split}: no _annotations.coco.json")
            continue
        d = json.loads(ann.read_text(encoding="utf-8"))

        cats = {c["id"]: c["name"] for c in d.get("categories", [])}
        print(f"\n  {split}: categories in file = {cats}")
        dropped = [c for c in cats if c != keep_class_id]
        if dropped:
            print(f"    dropping category_id(s) {dropped} (Roboflow parent classes) "
                  f"-> YOLO class 0 = '{class_name}'")

        by_img = {}
        for a in d["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)

        oimg = out / split / "images"
        olab = out / split / "labels"
        if not dry_run:
            oimg.mkdir(parents=True, exist_ok=True)
            olab.mkdir(parents=True, exist_ok=True)

        n_split = 0
        for im in d["images"]:
            src_img = idir / im["file_name"]
            if not src_img.exists():
                stats["missing_image"] += 1
                continue
            W, H = float(im["width"]), float(im["height"])
            lines = []
            for a in by_img.get(im["id"], []):
                if a["category_id"] != keep_class_id:
                    continue
                x, y, bw, bh = a["bbox"]
                if bw <= 0 or bh <= 0:
                    stats["degenerate_box"] += 1
                    continue
                lines.append(f"0 {(x + bw / 2) / W:.6f} {(y + bh / 2) / H:.6f} "
                             f"{bw / W:.6f} {bh / H:.6f}")
                aspects.append(bw / bh)
                areas.append(bw * bh / (W * H))

            stats["images"] += 1
            n_img += 1
            n_split += 1
            stats["boxes"] += len(lines)
            if not lines:
                stats["image_without_boxes"] += 1
                continue
            if not dry_run:
                shutil.copy2(src_img, oimg / im["file_name"])
                (olab / (Path(im["file_name"]).stem + ".txt")).write_text(
                    "\n".join(lines) + "\n", encoding="utf-8")
        print(f"    {split}: {n_split} images")

    print(f"\ntotal {stats['images']} images, {stats['boxes']} boxes")
    if stats["image_without_boxes"]:
        print(f"  WARN {stats['image_without_boxes']} images have no '{class_name}' box -> not copied")
    if stats["missing_image"]:
        print(f"  WARN {stats['missing_image']} annotation rows had no image file")
    if stats["degenerate_box"]:
        print(f"  WARN {stats['degenerate_box']} degenerate boxes skipped")
    if aspects:
        aspects.sort()
        mid = len(aspects) // 2
        med = (aspects[mid] if len(aspects) % 2 else (aspects[mid - 1] + aspects[mid]) / 2)
        print(f"\nbox aspect w/h   median {med:.3f}  min {aspects[0]:.2f}  max {aspects[-1]:.2f}")
        print(f"wider than tall  {100 * sum(a > 1 for a in aspects) / len(aspects):.0f}%")
        import statistics
        print(f"box area / image median {statistics.median(areas):.4f}")
    print("\nSERVE reference (live pipeline, 38 plates): box aspect median 0.57, "
          "19% wider-than-tall, box height 0.873 of the char row.")

    if not dry_run:
        import yaml
        out.mkdir(parents=True, exist_ok=True)
        y = out / "data.yaml"
        y.write_text(yaml.dump({
            "train": str((out / "train" / "images").resolve()),
            "val": str((out / "valid" / "images").resolve()),
            "test": str((out / "test" / "images").resolve()),
            "nc": 1,
            "names": [class_name],
        }, sort_keys=False), encoding="utf-8")
        print(f"\nwrote {y}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", required=True, help="COCO export root")
    ap.add_argument("--out", required=True, help="YOLO output root")
    ap.add_argument("--class-id", type=int, default=1,
                    help="COCO category_id to KEEP (1 = 'Each Thai Charactor car plate')")
    ap.add_argument("--class-name", default="char", help="YOLO class name")
    ap.add_argument("--dry-run", action="store_true", default=False)
    a = ap.parse_args()
    convert(Path(a.src), Path(a.out), a.class_id, a.class_name, a.dry_run)