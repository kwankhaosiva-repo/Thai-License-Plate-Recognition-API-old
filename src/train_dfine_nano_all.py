"""
src/train_dfine_nano_all.py

Master sequential training pipeline — LibreYOLO family (D-FINE / PicoDet).

Trains any variant (dfine_nano, dfine_small, picodet_s, picodet_m) on any
task (plate = Model 1, components = Model 2, charbox = Model 3A,
lao_plate = Lao) — WITHOUT overwriting old models.

Naming contract (deterministic, collision-free):
    weights/<artifact>_<variant>[_<tag>].pt / .onnx
    runs/train_<variant>/<task>[_<tag>]/

Usage:
  # เทรนเรียงลำดับ: Model 1 (Plate) -> Model 2 (Components) -> Model 3A (Charbox) -> Lao Plate
  python src/train_dfine_nano_all.py --all --epochs 30 --batch 8

  # ไม่ให้ทับโมเดลเดิม: ใส่ --tag ชื่อเวอร์ชัน (จะได้ *_<tag>.pt แทนการ overwrite)
  python src/train_dfine_nano_all.py --all --epochs 30 --batch 8 --tag v2

  # เทรนหลาย variant ต่อ task (เช่น M1 จากทั้ง 4 ตัว):
  python src/train_dfine_nano_all.py --task plate --variant dfine_nano,dfine_small,picodet_s,picodet_m --tag v2

  # เทรนทุก task x ทุก variant (16 งาน — ใช้ --dry-run ดูก่อนได้):
  python src/train_dfine_nano_all.py --all-variants --all-tasks --tag v2 --dry-run

Overwrite protection:
  - ถ้าไฟล์ปลายทาง .pt มีอยู่แล้วและ "ไม่ได้" ใส่ --tag → ข้าม task นั้น (โมเดลเดิมรอดแน่นอน)
  - ใส่ --overwrite เพื่อบังคับแทนที่แบบตั้งใจ
"""

import argparse
import time
import resource

# Expand file descriptor limit on macOS/Linux to avoid [Errno 24] Too many open files
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
except Exception:
    pass

from train_libreyolo_task import VARIANTS, TASKS, train_libreyolo_task, target_paths


def select_jobs(args):
    """Return ordered list of (task, variant) pairs from CLI args."""
    tasks = list(TASKS.keys())
    if args.all_tasks or args.train_all:
        selected_tasks = tasks
    else:
        selected_tasks = []
        raw_keys = [k.strip().lower() for k in args.task.split(",") if k.strip()]
        # Pass 1: exact matches (so 'lao_plate' never grabs 'plate')
        unmatched = []
        for k in raw_keys:
            if k in tasks:
                if k not in selected_tasks:
                    selected_tasks.append(k)
            else:
                unmatched.append(k)
        # Pass 2: fuzzy fallback for the rest
        for k in unmatched:
            matched = next((t for t in tasks if k in t or t in k), None)
            if matched is None:
                print(f"⚠️ Unknown task: '{k}'. Valid: {tasks}")
            elif matched not in selected_tasks:
                selected_tasks.append(matched)

    if args.all_variants:
        variants = list(VARIANTS.keys())
    else:
        variants = []
        for raw in args.variant.split(","):
            k = raw.strip().lower()
            if not k:
                continue
            if k not in VARIANTS:
                print(f"⚠️ Unknown variant: '{k}'. Valid: {list(VARIANTS.keys())}")
            elif k not in variants:
                variants.append(k)

    return [(task, variant) for task in selected_tasks for variant in variants]


def main():
    parser = argparse.ArgumentParser(
        description="Master Training Pipeline — LibreYOLO family (D-FINE / PicoDet), any task x any variant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", "--model", dest="task", type=str, default="plate,components,charbox,lao_plate",
                        help="Tasks to train (comma-separated): plate, components, charbox, lao_plate (--model works as alias)")
    parser.add_argument("--variant", type=str, default="dfine_nano",
                        help="Model variant(s) (comma-separated): dfine_nano, dfine_small, picodet_s, picodet_m")
    parser.add_argument("--all", dest="train_all", action="store_true",
                        help="Train all 4 tasks with the selected variant(s)")
    parser.add_argument("--all-tasks", action="store_true", help="Shorthand for --task plate,components,charbox,lao_plate")
    parser.add_argument("--all-variants", action="store_true", help="Shorthand for --variant dfine_nano,dfine_small,picodet_s,picodet_m")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override training resolution (DANGER: must also be used at ONNX export + inference; default = variant's contract: 640 dfine / 416 picodet)")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--tag", type=str, default="",
                        help="Version suffix (e.g. --tag v2 → *_v2.pt) — ใช้เพื่อไม่ให้ทับโมเดลเดิม")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing untagged weights (default: skip)")
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be trained and where outputs go, then exit")
    args = parser.parse_args()

    jobs = select_jobs(args)
    if not jobs:
        print("❌ No valid (task, variant) combinations selected. Exiting.")
        return

    print("=" * 80)
    print("🔥  LibreYOLO (D-FINE / PicoDet) — TRAINING SEQUENCE")
    print(f"   Tasks            : {', '.join(t for t, _ in jobs)}")
    print(f"   Variants         : {', '.join(v for _, v in jobs)}")
    print(f"   Tag              : {args.tag or '(none — untagged output paths)'}")
    print(f"   Compute Device   : {args.device.upper()}")
    print(f"   Epochs / Job     : {args.epochs} | Batch: {args.batch} | Patience: {args.patience}")
    if args.imgsz:
        print(f"   ⚠️ Imgsz OVERRIDE : {args.imgsz} (bypasses variant contract)")
    print(f"   Export ONNX      : {args.export_onnx}")
    print(f"   Overwrite mode   : {'ON (จะแทนที่ของเดิม)' if args.overwrite else 'OFF (ของเดิมปลอดภัย — จะข้ามถ้ามีไฟล์อยู่แล้ว)'}")
    print("=" * 80)

    results = {}
    for i, (task, variant) in enumerate(jobs, 1):
        paths = target_paths(task, variant, args.tag)
        exists = paths["pt"].exists()
        skip = exists and not args.overwrite and not args.tag

        if args.dry_run:
            status = "SKIP (exists)" if skip else "TRAIN"
            results[f"{task} x {variant}"] = f"[{status}] → {paths['pt'].name}"
            continue

        print(f"\n>>> [{i}/{len(jobs)}] {task} × {variant} → {paths['pt'].name}")
        t_start = time.time()
        try:
            trained = train_libreyolo_task(
                task=task,
                variant=variant,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                patience=args.patience,
                device=args.device,
                export_onnx=args.export_onnx,
                tag=args.tag,
                overwrite=args.overwrite,
            )
            elapsed = (time.time() - t_start) / 60
            if trained:
                results[f"{task} x {variant}"] = f"✅ Done in {elapsed:.1f} min → {paths['pt'].name}"
                print(f">>> [{i}/{len(jobs)}] Finished {task} × {variant} in {elapsed:.1f} min.")
            else:
                results[f"{task} x {variant}"] = f"⏭️  Skipped (exists) — {paths['pt'].name}"
        except Exception as e:
            results[f"{task} x {variant}"] = f"❌ FAILED: {e}"
            print(f">>> [{i}/{len(jobs)}] ⚠️  FAILED: {task} × {variant} — {e}")
            print("    Continuing with remaining jobs...")

    print("\n" + "=" * 80)
    print("🎉  Training Sequence Finished" + (" (DRY RUN)" if args.dry_run else ""))
    print("-" * 80)
    for name, status in results.items():
        print(f"  {status}")
    print("=" * 80)


if __name__ == "__main__":
    main()
