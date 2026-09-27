"""
src/merge_harvest_v4.py — merge harvested v4 crops into the v2 GT dataset.

WHAT IT DOES (idempotent — safe to re-run)
  1. Fix harvest province folders missing the NN_ prefix (e.g. `ระนอง` ->
     `41_ระนอง`) using weights/province_map.json as the source of truth.
  2. Copy harvest files into the GT `all/` stores with an `h4_` filename
     prefix (NEVER overwrites original GT crops that share the same stem).
  3. Rebuild recognition_v2/manifest.csv = original GT manifest rows +
     merged harvest rows (task/label/label_id/key/rel_path columns kept
     compatible with src/split_recognition_dataset_v2.py).
  4. Run the leak-free re-split (train/valid) for both tasks via
     split_recognition_dataset_v2's own functions (same grouping by source
     key, same stratified assignment, same materialise).

SAFETY
  - Original GT crops are never modified or overwritten (`h4_` prefix).
  - Backs up the GT-only manifest to manifest.gt_backup.csv on first run.
  - --dry-run shows the merge plan without copying anything.

USAGE
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/merge_harvest_v4.py            # merge + resplit
  ... --dry-run                                                                            # plan only
  ... --no-resplit                                                                         # merge only
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RECOG = PROJECT_ROOT / "datasets" / "Thai" / "recognition_v2"
HARVEST = {
    "char": RECOG / "char_crops",
    "province": RECOG / "province_crops",
}
ALL_DIR = {
    "char": RECOG / "char_crops_v2" / "all",
    "province": RECOG / "province_crops_v2" / "all",
}
MANIFEST_PATH = RECOG / "manifest.csv"
GT_BACKUP = RECOG / "manifest.gt_backup.csv"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map.json"

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_prov_map() -> dict[str, tuple[int, str]]:
    """thai_name -> (id, thai_name)"""
    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        pm = json.load(f)
    return {v: (int(k), v) for k, v in pm.items()}


def fix_province_folders(prov_by_name: dict) -> int:
    """Rename harvest province folders missing the NN_ prefix. Returns count fixed."""
    fixed = 0
    base = HARVEST["province"]
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if d.name.split("_")[0].isdigit():
            continue  # already has the prefix
        hit = prov_by_name.get(d.name)
        if hit is None:
            print(f"  !! unknown province folder (left as-is): {d.name}")
            continue
        pid, name = hit
        target = base / f"{pid:02d}_{name}"
        if target.exists():
            # merge contents then remove the prefix-less dir
            for f in d.iterdir():
                shutil.move(str(f), str(target / f.name))
            d.rmdir()
        else:
            d.rename(target)
        print(f"  renamed: {d.name} -> {target.name}")
        fixed += 1
    return fixed


def merge_task(task: str, dry_run: bool) -> list[dict]:
    """Copy harvest files into the GT all/ store with an h4_ prefix.
    Returns manifest rows for the merged crops."""
    src_root, dst_root = HARVEST[task], ALL_DIR[task]
    rows: list[dict] = []
    n_copy = n_skip = 0
    for cls_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        cls = cls_dir.name
        # province label_id from the folder prefix; char classes have none
        lid = ""
        if task == "province" and cls.split("_")[0].isdigit():
            lid = int(cls.split("_")[0])
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            dst_name = f"h4_{f.name}" if not f.name.startswith("h4_") else f.name
            # key = scene grouping id; harvest keys already contain the source
            # stem, but prefix with h4 so they never collide with GT keys
            key = f"h4_{f.stem}"
            dst = dst_root / cls / dst_name
            rel = f"{ALL_DIR[task].name.rsplit('_crops', 1)[0]}_crops_v2/all/{cls}/{dst_name}"
            rel = f"{'char' if task == 'char' else 'province'}_crops_v2/all/{cls}/{dst_name}"
            if dst.exists():
                n_skip += 1  # idempotent re-run
            elif not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n_copy += 1
            rows.append({
                "task": task,
                "key": key,
                "source_split": "harvest_v4",
                "label": cls if task == "char" or "_" not in cls else cls.split("_", 1)[1],
                "label_id": lid,
                "rel_path": rel,
                "crop_w": "",
                "crop_h": "",
                "m2_prov_conf": "",
                "m3a_conf": "",
                "label_source": "m3a_selflabel",
                "ocr_pseudo": "",
            })
    print(f"  {task}: copy {n_copy}, already-present {n_skip}, rows {len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Merge harvest_v4 crops into recognition_v2 + re-split")
    ap.add_argument("--dry-run", action="store_true", help="Show the plan, copy nothing")
    ap.add_argument("--no-resplit", action="store_true", help="Merge only, skip the leak-free re-split")
    args = ap.parse_args()

    prov_by_name = load_prov_map()

    print("== 1) Fix province folders missing NN_ prefix ==")
    if not args.dry_run:
        fix_province_folders(prov_by_name)
    else:
        base = HARVEST["province"]
        for d in sorted(base.iterdir()):
            if d.is_dir() and not d.name.split("_")[0].isdigit():
                hit = prov_by_name.get(d.name)
                print(f"  would rename: {d.name} -> {f'{hit[0]:02d}_{hit[1]}' if hit else '?? unknown'}")

    print("== 2) Merge harvest crops into GT all/ stores ==")
    new_rows = []
    for task in ("char", "province"):
        if HARVEST[task].is_dir():
            new_rows.extend(merge_task(task, args.dry_run))
        else:
            print(f"  {task}: harvest dir missing, skipped")

    if args.dry_run:
        print("\n(dry-run — nothing written)")
        return

    print("== 3) Rebuild manifest.csv (GT + harvest) ==")
    if not GT_BACKUP.exists():
        shutil.copy2(MANIFEST_PATH, GT_BACKUP)
        print(f"  backed up original manifest -> {GT_BACKUP.name}")
    with open(GT_BACKUP, "r", encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))
    gt_keys = {(r.get("task"), r.get("key"), r.get("rel_path")) for r in gt_rows}
    combined = list(gt_rows)
    added = 0
    for r in new_rows:
        if (r["task"], r["key"], r["rel_path"]) in gt_keys:
            continue
        combined.append(r)
        added += 1
    fieldnames = []
    for r in combined:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(MANIFEST_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(combined)
    print(f"  manifest rows: GT {len(gt_rows)} + harvest {added} = {len(combined)}")

    if args.no_resplit:
        print("== 4) re-split skipped (--no-resplit) ==")
        return

    print("== 4) Leak-free re-split (train/valid) ==")
    from src.split_recognition_dataset_v2 import (
        group_by_class_and_key, stratified_assign, report, materialise, write_reports,
    )
    for task in ("province", "char"):
        rows = [r for r in combined if r.get("task") == task]
        if not rows:
            print(f"  {task}: no rows — skipping")
            continue
        by_class = group_by_class_and_key(rows)
        assignment, stats = stratified_assign(by_class, 0.2, 42)
        report(task, stats, assignment, by_class)
        moved = materialise(task, rows, assignment, use_symlink=False)
        rep, man = write_reports(task, stats, rows, assignment)
        tr = sum(v for (s, _), v in moved.items() if s == "train")
        va = sum(v for (s, _), v in moved.items() if s == "valid")
        print(f"  {task}: materialised train {tr} / valid {va} (copied, not symlinked)")

    print("\nDone. Next: scratch/audit_v4_balance.py for size/aspect/contrast/balance check.")


if __name__ == "__main__":
    main()
