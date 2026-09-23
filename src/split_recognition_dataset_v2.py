"""
src/split_recognition_dataset_v2.py

STEP 2 of the v2 recognition retrain — build a LEAK-FREE train/valid split.

WHY A NEW SPLITTER
==================
The audits (scratch/audit_split_leakage.py, scratch/audit_phash_leakage.py)
measured real leakage in the datasets currently on disk:

  * M3A charbox split : 8 timestamps (identical capture second -> same vehicle
                        from the same video) appear in BOTH train and valid.
  * M3B province split: 20 crops have the byte-identical filename in train AND
                        valid, because extract_thai_province_crops.py copies
                        seeds from valid/test into train and then runs
                        augment_real_province_crop() on them
                        (train holds 9120 crops: 5935 aug_real_ + 412 aug_truck_
                         + 2345 rf_ + 428 gt_).
  * M3A char split    : merge_candidates_to_dataset.py assigns train/valid with
                        np.random.rand() PER CROP, so crops of one plate land in
                        both splits.

Inflated validation accuracy cannot be compared against a v2 model, so the v2
split is assigned at the SOURCE-IMAGE level (the `key` column of manifest.csv):
all crops derived from one source image go to exactly one split.

Method: greedy stratified assignment. Classes are processed rarest-first; each
source image is placed so the per-class valid ratio stays as close to
--valid-ratio as possible. Deterministic (seeded), so it is reproducible.

Usage
=====
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/split_recognition_dataset_v2.py
  ... --valid-ratio 0.2 --seed 42
  ... --task province          # only materialise the province split
  ... --task char
  ... --dry-run                # report only, write nothing

Output under datasets/Thai/recognition_v2/<task>_crops_v2/
  all/<class>/...            (extraction output, kept for reference)
  train/<class>/...          symlinks to all/... (no data duplication)
  valid/<class>/...
  split_report.csv           per-class train/valid counts
  split_manifest.csv         per-crop split assignment
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECOG_DIR = PROJECT_ROOT / "datasets" / "Thai" / "recognition_v2"
MANIFEST_PATH = RECOG_DIR / "manifest.csv"

TASK_DIRS = {
    "province": "province_crops_v2",
    "char": "char_crops_v2",
    "char_row": "char_row_crops_v2",
}


def load_manifest(task: str) -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"manifest not found: {MANIFEST_PATH}\n"
            "run src/extract_recognition_crops_v2.py first"
        )
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("task") == task:
                rows.append(r)
    return rows


def group_by_class_and_key(rows: list[dict]):
    """class -> key -> [rows]  (a key = one source image; never split)."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["label"]][r["key"]].append(r)
    return out


def stratified_assign(by_class, valid_ratio: float, seed: int):
    """Assign WHOLE source images to train/valid. Returns (key -> split, stats).

    IMPORTANT: the unit of assignment is the *source image* (`key`), NOT the
    (class, key) pair. One plate contributes character crops of several classes
    (e.g. `000002_0004` -> 2, 8, 2, 0, 4, 1), so assigning per (class, key) would
    let the same source image land in train via one character and in valid via
    another — exactly the leakage this script exists to remove.

    Method: each class gets a valid-quota of valid_ratio * its crop count.
    Source images are visited rarest-class-first (shuffled within a rarity tier
    for determinism); a key goes to `valid` only while it still feeds an unmet
    quota, and at least one key is always left for train.
    """
    key_labels: dict[str, set] = defaultdict(set)
    for cls, keys in by_class.items():
        for k in keys:
            key_labels[k].add(cls)

    class_total = {c: sum(len(by_class[c][k]) for k in by_class[c]) for c in by_class}
    quota = {c: int(round(t * valid_ratio)) for c, t in class_total.items()}

    rng = random.Random(seed)
    tiers: dict[int, list[str]] = defaultdict(list)
    for k, labels in key_labels.items():
        tiers[min(class_total[c] for c in labels)].append(k)
    ordered: list[str] = []
    for rarity in sorted(tiers):
        bucket = sorted(tiers[rarity])
        rng.shuffle(bucket)
        ordered.extend(bucket)

    taken = {c: 0 for c in by_class}
    keys_left = {c: len(by_class[c]) for c in by_class}
    global_budget = max(0, int(round(len(ordered) * valid_ratio)))
    assignment: dict[str, str] = {}
    n_valid = 0
    for k in ordered:
        labels = sorted(key_labels[k])
        # Every class sharing this source image must still keep >=1 key for train.
        keeps_train = all(keys_left[c] > 1 for c in labels)
        gains = sum(min(len(by_class[c][k]), max(0, quota[c] - taken[c])) for c in labels)
        if keeps_train and gains > 0 and n_valid < min(global_budget, len(ordered) - 1):
            for c in labels:
                taken[c] += len(by_class[c][k])
            assignment[k] = "valid"
            n_valid += 1
        else:
            assignment[k] = "train"
        for c in labels:
            keys_left[c] -= 1


    stats = {}
    for cls, keys in by_class.items():
        vc = sum(len(by_class[cls][k]) for k in keys if assignment[k] == "valid")
        stats[cls] = {
            "groups": len(keys),
            "crops": class_total[cls],
            "valid_groups": sum(1 for k in keys if assignment[k] == "valid"),
            "valid_crops": vc,
            "train_crops": class_total[cls] - vc,
        }
    return assignment, stats



def class_dir_name(r: dict) -> str:
    """Folder name for the materialised split.

    The Thai province trainer (train_grayscale_province_thai.py) resolves a
    class from the folder name `NN_thai_name`, and the char trainer
    (train_character_classifier.py) uses the bare character. Keep both
    conventions so the v2 splits are drop-in compatible with the trainers.
    """
    try:
        lid = int(r.get("label_id") or -1)
    except (TypeError, ValueError):
        lid = -1
    return f"{lid:02d}_{r['label']}" if lid >= 0 else r["label"]


def materialise(task: str, rows, assignment, use_symlink: bool = True):
    """Create <task>_crops_v2/{train,valid}/<class>/... from the `all/` store."""
    base = RECOG_DIR / TASK_DIRS[task]
    moved = Counter()
    for r in rows:
        cls_dir, key, rel = class_dir_name(r), r["key"], r["rel_path"]
        split = assignment[key]
        src = RECOG_DIR / rel
        if not src.exists():
            continue
        dst = base / split / cls_dir / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if use_symlink:
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        moved[(split, cls_dir)] += 1
    return moved


def write_reports(task: str, stats, rows, assignment):
    base = RECOG_DIR / TASK_DIRS[task]
    base.mkdir(parents=True, exist_ok=True)

    rep = base / "split_report.csv"
    with open(rep, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "groups", "crops", "train_crops",
                                          "valid_crops", "valid_groups", "valid_ratio"])
        w.writeheader()
        for cls, s in sorted(stats.items(), key=lambda kv: -(kv[1]["train_crops"] + kv[1]["valid_crops"])):
            w.writerow({
                "label": cls,
                **s,
                "valid_ratio": round(s["valid_crops"] / max(s["crops"], 1), 4),
            })

    man = base / "split_manifest.csv"
    with open(man, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "key", "label", "split",
                                          "source_split", "rel_path",
                                          "m2_prov_conf", "m3a_conf", "gt_text"])
        w.writeheader()
        for r in rows:
            cls, key = r["label"], r["key"]
            w.writerow({
                "task": task,
                "key": key,
                "label": cls,
                "split": assignment[key],
                "source_split": r.get("source_split", ""),
                "rel_path": r["rel_path"],
                "m2_prov_conf": r.get("m2_prov_conf", ""),
                "m3a_conf": r.get("m3a_conf", ""),
                "gt_text": r.get("gt_text", ""),
            })
    return rep, man


def report(task: str, stats, assignment, by_class):
    tot_crops = sum(s["crops"] for s in stats.values())
    tot_valid = sum(s["valid_crops"] for s in stats.values())
    print(f"\n### {task}")
    print(f"  classes            : {len(stats)}")
    print(f"  source images      : {sum(s['groups'] for s in stats.values())}")
    print(f"  crops              : {tot_crops}  (train {tot_crops - tot_valid} / valid {tot_valid})")
    print(f"  overall valid ratio: {tot_valid / max(tot_crops, 1):.3f}")
    thin = [(c, s) for c, s in stats.items() if s["train_crops"] == 0]
    if thin:
        print(f"  !! classes with ZERO train crops: {[c for c, _ in thin]}")
    thin_v = [(c, s) for c, s in stats.items() if s["valid_crops"] == 0]
    if thin_v:
        print(f"  note: {len(thin_v)} classes have no valid crop (single source image): "
              f"{[c for c, _ in thin_v][:10]}")

    # leakage self-check: no source image may span both splits
    keys_by_split = defaultdict(set)
    for key, sp in assignment.items():
        keys_by_split[sp].add(key)
    shared = keys_by_split["train"] & keys_by_split["valid"]
    print(f"  leakage self-check : source images spanning both splits = {len(shared)} "
          f"({'OK' if not shared else 'FAIL'})")



def main():
    ap = argparse.ArgumentParser(description="Leak-free v2 train/valid split")
    ap.add_argument("--valid-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", choices=["province", "char", "all"], default="province")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = ["province", "char"] if args.task == "all" else [args.task]
    for task in tasks:
        rows = load_manifest(task)
        if not rows:
            print(f"\n### {task}: manifest has no '{task}' rows — skipping")
            continue
        by_class = group_by_class_and_key(rows)
        assignment, stats = stratified_assign(by_class, args.valid_ratio, args.seed)
        report(task, stats, assignment, by_class)
        if args.dry_run:
            continue
        # rebuild the materialised split from scratch (drops stale class dirs
        # from an older run, e.g. folders named without the NN_ prefix)
        base = RECOG_DIR / TASK_DIRS[task]
        for stale in ("train", "valid"):
            shutil.rmtree(base / stale, ignore_errors=True)
        moved = materialise(task, rows, assignment, use_symlink=not args.copy)
        rep, man = write_reports(task, stats, rows, assignment)
        print(f"  materialised       : {sum(moved.values())} links")
        print(f"  report             : {rep.relative_to(PROJECT_ROOT)}")
        print(f"  split manifest     : {man.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()


