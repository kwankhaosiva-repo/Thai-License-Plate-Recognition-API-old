"""
src/curate_recognition_v2.py

SAFE curation of recognition_v2 crops (label corrections).

WHY THIS EXISTS
===============
train/ and valid/ contain symlinks into all/. Moving or deleting files through
train/ or valid/ is unsafe:

  * deleting a link deletes the real file in all/  (data loss)
  * moving a link moves only the link — the real file stays in the ORIGINAL
    class folder in all/ and manifest.csv still records the old label, so the
    correction never reaches training.

The correct workflow is to operate on the REAL file in all/ and update
manifest.csv in the same step, then rebuild the split:

  python src/split_recognition_dataset_v2.py --task all --seed 42
  python scratch/audit_recognition_v2.py

Usage
=====
  # locate a file by (partial) name, across both tasks
  python src/curate_recognition_v2.py find --name 20200213_094055

  # move a crop to its correct class (province classes are 'NN_thai', char is bare char)
  python src/curate_recognition_v2.py move --name 20200213_094055_jpg.jpg --to 32_พังงา
  python src/curate_recognition_v2.py move --name 000096_0319_plate_jpg_b2.jpg --to ฑ

  # delete a bad crop (removes file + manifest rows)
  python src/curate_recognition_v2.py delete --name 20200213_094055_jpg.jpg --yes

  # restore a previously deleted crop from its original source store
  # (plate_crops_v2 / char_row_crops_v2 keep the uncropped originals)
  python src/curate_recognition_v2.py restore --name 20200213_094055_jpg.jpg

A timestamped backup of manifest.csv is written to <recognition_v2>/manifest_backups/
before every write.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOG_DIR = PROJECT_ROOT / "datasets" / "Thai" / "recognition_v2"
MANIFEST_PATH = RECOG_DIR / "manifest.csv"
BACKUP_DIR = RECOG_DIR / "manifest_backups"
TASK_DIRS = {
    "province": "province_crops_v2",
    "char": "char_crops_v2",
}
IMG_EXT = {".jpg", ".jpeg", ".png"}
MANIFEST_FIELDS = ["task", "key", "source_split", "label", "label_id", "rel_path",
                   "crop_w", "crop_h", "m2_prov_conf", "province_box_source",
                   "m3a_conf", "char_index", "gt_text"]


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_manifest(rows: list[dict]) -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / f"manifest_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy2(MANIFEST_PATH, backup)
    with open(MANIFEST_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  manifest saved (backup: {backup.relative_to(RECOG_DIR)})")


def rows_matching(rows: list[dict], name: str) -> list[dict]:
    """Manifest rows whose rel_path basename matches exactly, or uniquely as a substring."""
    exact = [r for r in rows if Path(r["rel_path"]).name == name]
    if exact:
        return exact
    hits = [r for r in rows if name in Path(r["rel_path"]).name]
    return hits


def find_on_disk(rel_path: str) -> Path | None:
    """The real file may have been moved within all/; search the task's all/ tree."""
    task_dir = RECOG_DIR / rel_path.split("/")[0] / "all"
    base = Path(rel_path).name
    hits = [p for p in task_dir.rglob(base) if p.is_file()]
    return hits[0] if len(hits) == 1 else (hits[0] if hits else None)


def class_label_from_dir(class_dir: str, task: str) -> tuple[str, str]:
    """Return (label, label_id) implied by a class folder name.

    province folders are 'NN_thai_name' (label = thai name, id = NN);
    char folders are the bare character (label_id stays -1).
    """
    if task == "province" and "_" in class_dir:
        prefix, _, rest = class_dir.partition("_")
        if prefix.isdigit():
            return rest, prefix
    return class_dir, "-1"


def cmd_find(args) -> None:
    rows = load_manifest()
    hits = rows_matching(rows, args.name)
    if not hits:
        print(f"no manifest row matches '{args.name}'")
        return
    for r in hits:
        exists = (RECOG_DIR / r["rel_path"]).exists()
        print(f"[{r['task']}] {Path(r['rel_path']).name}")
        print(f"    label={r['label']}  rel={r['rel_path']}  on_disk={'YES' if exists else 'NO'}")


def cmd_move(args) -> None:
    rows = load_manifest()
    hits = rows_matching(rows, args.name)
    if not hits:
        print(f"no manifest row matches '{args.name}'")
        return
    if len(hits) > 1:
        print(f"ambiguous '{args.name}' matches {len(hits)} rows — use a longer name:")
        for r in hits:
            print(f"    {Path(r['rel_path']).name}  (label={r['label']})")
        return

    r = hits[0]
    task = r["task"]
    task_dir = RECOG_DIR / TASK_DIRS[task]
    dst_class_dir = task_dir / "all" / args.to
    if not dst_class_dir.is_dir():
        print(f"target class folder not found: {dst_class_dir}")
        return

    src = find_on_disk(r["rel_path"])
    if src is None:
        print(f"file not found on disk: {r['rel_path']}")
        return
    dst = dst_class_dir / src.name
    if dst.exists():
        print(f"destination already exists: {dst}")
        return

    new_label, new_id = class_label_from_dir(args.to, task)
    shutil.move(str(src), str(dst))
    r["rel_path"] = dst.relative_to(RECOG_DIR).as_posix()
    r["label"] = new_label
    if new_id != "-1":
        r["label_id"] = new_id
    save_manifest(rows)
    print(f"  moved {src.relative_to(RECOG_DIR)}")
    print(f"     -> {r['rel_path']}  label={new_label}")
    print("\nnext: python src/split_recognition_dataset_v2.py --task all --seed 42")


def cmd_delete(args) -> None:
    rows = load_manifest()
    hits = rows_matching(rows, args.name)
    if not hits:
        print(f"no manifest row matches '{args.name}'")
        return
    if len(hits) > 1 and not args.yes:
        print(f"'{args.name}' matches {len(hits)} rows — re-run with --yes to delete all:")
        for r in hits:
            print(f"    {r['rel_path']}  (label={r['label']})")
        return
    for r in hits:
        p = RECOG_DIR / r["rel_path"]
        if p.exists():
            p.unlink()
            print(f"  deleted file: {r['rel_path']}")
        else:
            print(f"  (already absent): {r['rel_path']}")
    remaining = [x for x in rows if x not in hits]
    print(f"  removed {len(rows) - len(remaining)} manifest rows")
    save_manifest(remaining)
    print("\nnext: python src/split_recognition_dataset_v2.py --task all --seed 42")


def cmd_restore(args) -> None:
    """Recreate a deleted crop from plate_crops_v2 / char_row_crops_v2 originals."""
    rows = load_manifest()
    backup_hits = []
    for b in sorted(BACKUP_DIR.glob("manifest_*.csv")):
        with open(b, encoding="utf-8-sig", newline="") as f:
            backup_hits += [r for r in csv.DictReader(f) if args.name in Path(r["rel_path"]).name]
    if not backup_hits:
        print(f"no backup manifest row matches '{args.name}'")
        return
    for r in backup_hits:
        print(f"  backup row: [{r['task']}] {r['rel_path']}  label={r['label']}")
    # locate the original in plate_crops_v2/char_row_crops_v2 (same basename, same class)
    for r in backup_hits:
        base = Path(r["rel_path"]).name
        class_dir = Path(r["rel_path"]).parts[-2]
        for store in ("plate_crops_v2", "char_row_crops_v2"):
            candidate = RECOG_DIR / store / "all" / class_dir / base
            if candidate.exists():
                dst = RECOG_DIR / r["rel_path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, dst)
                rows.append(r)
                print(f"  restored {dst.relative_to(RECOG_DIR)}  <-  {candidate.relative_to(RECOG_DIR)}")
                break
        else:
            print(f"  !! no original found for {base} in plate_crops_v2/char_row_crops_v2")
    save_manifest(rows)
    print("\nnext: python src/split_recognition_dataset_v2.py --task all --seed 42")


def main() -> None:
    ap = argparse.ArgumentParser(description="Safely curate recognition_v2 crops (all/ + manifest in sync)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("find", help="locate a crop by name")
    p.add_argument("--name", required=True)
    p.set_defaults(fn=cmd_find)

    p = sub.add_parser("move", help="move a crop to its correct class folder")
    p.add_argument("--name", required=True, help="file basename (or unique substring)")
    p.add_argument("--to", required=True, help="target class folder name, e.g. 32_พังงา or ฑ")
    p.set_defaults(fn=cmd_move)

    p = sub.add_parser("delete", help="delete a crop (file + manifest rows)")
    p.add_argument("--name", required=True)
    p.add_argument("--yes", action="store_true", help="apply even if the name matches several rows")
    p.set_defaults(fn=cmd_delete)

    p = sub.add_parser("restore", help="restore a deleted crop from its source store")
    p.add_argument("--name", required=True)
    p.set_defaults(fn=cmd_restore)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
