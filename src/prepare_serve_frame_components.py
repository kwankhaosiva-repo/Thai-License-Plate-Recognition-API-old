"""
src/prepare_serve_frame_components.py

Rebuild the Model-2 (component detector) training set in the EXACT plate frame
that serve feeds it, from the v5i province labels.

WHY THIS EXISTS
===============
`datasets/Thai/LPR 2 - Charactor Detection.yolov11` (what M2 is trained on) is
drawn on Roboflow crops whose plate layout differs from the plate layout serve
hands M2. Measured on 250 images per set (scratch/audit_retrain_datasets.py):

    province box, as a fraction of the plate image
        train  (LPR 2 - Charactor Detection, 162x84)   cy 0.697   h 0.186   aspect 5.72
        serve  (rectified plate, 320x160)               cy 0.856   h 0.225   aspect 3.40

    char-row box, as a fraction of the plate image
        train                                                cy 0.385   h 0.381   aspect 3.83
        serve                                                cy 0.419   h 0.531   aspect 3.22

The province box M2 predicts at serve sits **0.16 of the plate height lower**
than any province box it was trained on. The model still works (measured IoU vs
the v5i ground truth: median 0.798) because it generalises onto the v5i/v5
layout — but it is being asked to regress a coordinate frame it never saw, which
is exactly the kind of "preprocessing/model must compensate" situation this
project is trying to remove.

v5i (`thai-car-license-plate-province.v5i.yolov11`) is the one dataset that IS
in the serve frame: its human province boxes sit at cy 0.85 / h 0.24 of a
192x99 rectified plate. So it is the correct source. This script warps each v5i
plate through the SAME rectification serve uses and remaps the province box
through the exact perspective matrix.

THE RECONSTRUCTION IS EXACT FOR THE PROVINCE BOX
================================================
`warp_perspective_plate` is the single implementation used at serve
(src/api_server.py) and here, and the corner->pixel mapping is the same
`cv2.getPerspectiveTransform`, so the remapped box is the box serve would see.
`warp_perspective_plate` does not expose its matrix, so this script rebuilds it
with the identical `order_points_clockwise` + padding expansion; a unit check
against the serve function is in src/test_preprocess_registry.py.

THE CHAR-ROW BOX IS DERIVED, NOT ANNOTATED
===========================================
v5i carries no per-character boxes. The char-row box is derived from the
province box using the serve crop gates, and is therefore only as good as those
gates. It lands at cy 0.345 / h 0.65 / aspect 2.92 against the serve-measured
cy 0.419 / h 0.531 / aspect 3.22 — close, but a genuine annotation gap. Keep
`LPR 2 - Character Box Detection.yolov11` as the M3A source, and re-measure M2
after training before retiring the old set.

USAGE
=====
  # dry run, report only
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/prepare_serve_frame_components.py

  # build
  ... --out "datasets/Thai/LPR2_ServeFrame.yolov11"

  # re-use the existing v5i train/valid/test split (no new leakage, same grouping)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_perspective_dataset import (  # noqa: E402
    order_points_clockwise,
    warp_perspective_plate,
)

SRC = PROJECT_ROOT / "datasets" / "Thai" / "thai-car-license-plate-province.v5i.yolov11"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "Thai" / "LPR2_ServeFrame.yolov11"

# Must match api_server.py: the rectified plate is 320x160 with padding_frac 0.06
# (the plate_corner_regressor branch is the one that actually runs — LibreYOLOWrapper
# forces r.masks = None, so the polygon branch never executes).
PLATE_W, PLATE_H = 320, 160
PADDING_FRAC = 0.06

# Serve crop gates (src/api_server.py, M2 branch). Used to DERIVE the char-row box
# from the annotated province box; see the module docstring for the caveat.
CHAR_ROW_TOP_FRAC = 0.02
CHAR_ROW_BOTTOM_CLAMP = 0.68
CHAR_ROW_X0_FRAC = 0.03
CHAR_ROW_X1_FRAC = 0.97

CLASS_CHAR_ROW = 0
CLASS_PROVINCE = 1
NAMES = ["plate_char", "province"]


def serve_perspective_matrix(quad: np.ndarray) -> np.ndarray:
    """Rebuild the exact matrix `warp_perspective_plate` uses for this quad.

    Mirrors prepare_perspective_dataset.warp_perspective_plate step for step:
    order corners clockwise, expand outward by padding_frac, map onto the
    [0,calc_w-1] x [0,calc_h-1] destination rectangle.
    """
    tl, tr, br, bl = order_points_clockwise(quad)

    cx = float(np.mean([tl[0], tr[0], br[0], bl[0]]))
    cy = float(np.mean([tl[1], tr[1], br[1], bl[1]]))
    expanded = []
    for pt in (tl, tr, br, bl):
        dx, dy = pt[0] - cx, pt[1] - cy
        norm = max(float(np.sqrt(dx ** 2 + dy ** 2)), 1e-6)
        pad = PADDING_FRAC * max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(bl - tl)))
        expanded.append([pt[0] + dx / norm * pad, pt[1] + dy / norm * pad])
    src = np.array(expanded, dtype=np.float32)
    dst = np.array(
        [[0, 0], [PLATE_W - 1, 0], [PLATE_W - 1, PLATE_H - 1], [0, PLATE_H - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst), src, dst


def map_points(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a perspective matrix to an (N,2) array of points."""
    p = np.hstack([pts.reshape(-1, 2), np.ones((len(pts), 1), dtype=np.float32)])
    out = p @ matrix.T
    w = np.where(np.abs(out[:, 2:3]) < 1e-9, 1e-9, out[:, 2:3])
    return out[:, :2] / w


def derive_char_row(prov_y1: float, prov_y2: float) -> tuple[float, float]:
    """Char-row band implied by the annotated province box + the serve gates.

    Serve keeps the char row while its top is above 0.65*rh and clamps the crop
    bottom to 0.68*rh, so the band is [top margin .. province top - gap].
    """
    y1 = CHAR_ROW_TOP_FRAC * PLATE_H
    y2 = min(CHAR_ROW_BOTTOM_CLAMP * PLATE_H, prov_y1 - 0.06 * PLATE_H)
    return y1, max(y2, y1 + 8.0)


def build(dry_run: bool, out_dir: Path, limit: int | None):
    if not SRC.exists():
        raise SystemExit(f"source dataset missing: {SRC}")

    if dry_run:
        print("DRY RUN — nothing written\n")
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for split in ("train", "valid", "test"):
            (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    grand, stats = Counter(), {"prov_cy": [], "prov_h": [], "char_cy": [], "char_h": [],
                               "char_asp": [], "prov_asp": []}
    total = skipped = 0

    for split in ("train", "valid", "test"):
        idir = SRC / split / "images"
        if not idir.exists():
            continue
        files = sorted(f for f in idir.iterdir()
                       if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if limit:
            files = files[:limit]
        n_split = 0

        for p in files:
            lab = SRC / split / "labels" / (p.stem + ".txt")
            if not lab.exists():
                skipped += 1
                continue
            parts = lab.read_text(encoding="utf-8").split()
            if len(parts) < 5:
                skipped += 1
                continue

            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = (float(v) for v in parts[1:5])
            img = cv2.imread(str(p))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]

            # Source province box in source pixels.
            px1, py1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            px2, py2 = (xc + bw / 2) * w, (yc + bh / 2) * h

            # Rectify through the serve warp, then map the box with the SAME matrix.
            quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            matrix, _, _ = serve_perspective_matrix(quad)
            plate = warp_perspective_plate(img, quad, target_width=PLATE_W,
                                           target_height=PLATE_H, padding_frac=PADDING_FRAC)
            corners = map_points(matrix, np.array([[px1, py1], [px2, py1],
                                                   [px2, py2], [px1, py2]], dtype=np.float32))
            nx1, ny1 = corners[:, 0].min(), corners[:, 1].min()
            nx2, ny2 = corners[:, 0].max(), corners[:, 1].max()
            nx1, nx2 = np.clip([nx1, nx2], 0, PLATE_W)
            ny1, ny2 = np.clip([ny1, ny2], 0, PLATE_H)
            if nx2 - nx1 < 8 or ny2 - ny1 < 5:
                skipped += 1
                continue

            cy1, cy2 = derive_char_row(ny1, ny2)
            cx1, cx2 = CHAR_ROW_X0_FRAC * PLATE_W, CHAR_ROW_X1_FRAC * PLATE_W

            stats["prov_cy"].append((ny1 + ny2) / 2 / PLATE_H)
            stats["prov_h"].append((ny2 - ny1) / PLATE_H)
            stats["prov_asp"].append((nx2 - nx1) / max(ny2 - ny1, 1e-6))
            stats["char_cy"].append((cy1 + cy2) / 2 / PLATE_H)
            stats["char_h"].append((cy2 - cy1) / PLATE_H)
            stats["char_asp"].append((cx2 - cx1) / max(cy2 - cy1, 1e-6))

            grand[cls_id] += 1
            total += 1
            n_split += 1

            if dry_run:
                continue

            cv2.imwrite(str(out_dir / split / "images" / f"{p.name}"),
                        plate, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            def yolo(cls, x1, y1, x2, y2):
                return (f"{cls} {(x1 + x2) / 2 / PLATE_W:.6f} {(y1 + y2) / 2 / PLATE_H:.6f} "
                        f"{(x2 - x1) / PLATE_W:.6f} {(y2 - y1) / PLATE_H:.6f}")

            (out_dir / split / "labels" / f"{p.stem}.txt").write_text(
                yolo(CLASS_CHAR_ROW, cx1, cy1, cx2, cy2) + "\n" +
                yolo(CLASS_PROVINCE, nx1, ny1, nx2, ny2) + "\n", encoding="utf-8")

        print(f"  {split:6} {n_split:5d} images")

    if not dry_run:
        yaml_path = out_dir / "data.yaml"
        yaml_path.write_text(yaml.dump({
            "train": str((out_dir / "train" / "images").resolve()),
            "val": str((out_dir / "valid" / "images").resolve()),
            "test": str((out_dir / "test" / "images").resolve()),
            "nc": 2,
            "names": NAMES,
        }, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"\nwrote {yaml_path}")

    print(f"\ntotal {total} plates, skipped {skipped}, {len(grand)} province classes")
    print("\nframe check (fraction of the 320x160 serve plate):")
    for k, label in (("prov_cy", "province cy"), ("prov_h", "province h"),
                     ("prov_asp", "province aspect"), ("char_cy", "char-row cy"),
                     ("char_h", "char-row h"), ("char_asp", "char-row aspect")):
        v = np.array(stats[k])
        print(f"  {label:18} med={np.median(v):6.3f}  [p10={np.percentile(v,10):6.3f} "
              f"p90={np.percentile(v,90):6.3f}]")
    print("\nserve-measured targets: province cy 0.856 h 0.225 aspect 3.40 | "
          "char-row cy 0.419 h 0.531 aspect 3.22")
    print("old train frame:         province cy 0.697 h 0.186 aspect 5.72 | "
          "char-row cy 0.385 h 0.381 aspect 3.83")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output dataset dir")
    ap.add_argument("--limit", type=int, default=None, help="max images per split (smoke test)")
    ap.add_argument("--dry-run", action="store_true", default=False, help="report only, write nothing")
    args = ap.parse_args()
    build(args.dry_run, Path(args.out), args.limit)