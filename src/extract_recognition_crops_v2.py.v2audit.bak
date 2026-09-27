"""
src/extract_recognition_crops_v2.py

STEP 1 of the v2 recognition retrain — re-extract province / character / plate
crops with the CURRENT production pipeline (M2 -> M3A) so that the TRAIN crop
distribution is byte-for-byte the same *function* as the SERVE crop
distribution.

WHY A NEW EXTRACTION IS REQUIRED
================================
The old datasets were built out of Roboflow hand-drawn boxes:
  * province  -> thai-car-license-plate-province.v5i (box drawn by a human)
  * char cls  -> output/ground_truth_crops/gt_plate_char.csv (file deleted)
The crops the live pipeline produces are NOT that distribution: api_server.py
applies its own insets/padding/min-y gating (see CROP RULES below). Training on
human boxes and serving pipeline crops is exactly the train/serve mismatch that
makes the province model mis-read plates in production.

SOURCES (both carry a real label)
=================================
  A) datasets/Thai/thai-car-license-plate-province.v5i.yolov11/<split>/
       192x99 rectified PLATE crops (measured median aspect 1.94 == M2 train
       aspect 1.941). Label = 1 line "<cls_id> <xc yc w h>" for the province
       line. cls_id -> abbr (data.yaml names) -> Thai name
       (weights/province_abbr_map.json).
       => produces plate_crops_v2 (scene) + province_crops_v2
  B) datasets/Thai/LPR 2 - Character Box Detection.yolov11/<split>/
       ~216x62 CHARACTER-ROW crops (measured median aspect 3.712 == M3A train
       aspect). Label = one YOLO box per Thai character (class 0).
       Character TEXT comes from datasets/Thai/thai_character_crops/metadata.csv
       (columns tag_id / raw_gt_plate / box_index / character / source_image) —
       this CSV is the surviving copy of the deleted gt_plate_char.csv.
       => produces char_crops_v2

OUTPUT (datasets/Thai/recognition_v2/)
======================================
  manifest.csv                  every accepted crop + provenance + label
  rejected.csv                  every rejected crop + the exact reason
  plate_crops_v2/all/<NN_name>/      M2 rectified-plate ("scene") crop
  province_crops_v2/all/<NN_name>/   M2 province crop, SERVE padding applied
  char_crops_v2/all/<char>/          M3A per-character crop, square padded 64x64

Splitting is NOT done here — run src/split_recognition_dataset_v2.py afterwards
so the split can be grouped by source image (no leakage).

CROP RULES (must stay identical to api_server.py)
=================================================
  M2 province box  : api_server.py:1796-1813
      accept if by2 > 0.50*rh AND center_y > 0.45*rh
      pad_x = min(14, max(6, int(bw*0.08))), pad_y = min(8, max(4, int(bh*0.10)))
      px1 = max(0, bx1-pad_x); px2 = min(rw, bx2+pad_x)
      py1 = max(0.46*rh, by1-pad_y); py2 = min(rh, by2+pad_y)
  M2 char box      : api_server.py:1778-1794
      accept if by1 < 0.65*rh
      pad_cx = min(8, max(2, int(bw*0.03))), pad_cy = min(6, max(2, int(bh*0.04)))
      cx1 = max(0.03*rw, bx1-pad_cx); cx2 = min(0.97*rw, bx2+pad_cx)
      cy1 = max(0.02*rh, by1-pad_cy); cy2 = min(0.68*rh, by2+pad_cy)
  M3A char box     : api_server.py:1927-1949
      divide_boxes_by_scale -> charbox_greedy_nms(iou 0.35)
      -> split_wide_char_boxes -> drop bw<8 or bh<10
  char classifier  : extract_thai_character_crops.make_square_padded(64)
      (identical to the serve pad at api_server.py:2035-2041)

Usage
=====
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/extract_recognition_crops_v2.py
  ... --only province                    # just rebuild province crops
  ... --only char
  ... --limit 200                        # smoke test
  ... --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg  # noqa: E402
from src.preprocess_registry import (  # noqa: E402
    apply_override,
    divide_boxes_by_scale,
    get_m2_spec,
    get_m3a_spec,
    upscale_to_train_geometry,
)
from src.api_server import (  # noqa: E402  (only pure helpers are used)
    LibreYOLOWrapper,
    RFDETRWrapper,
    RTDETR,
    charbox_greedy_nms,
    split_wide_char_boxes,
)

# --- Paths -----------------------------------------------------------------
THAI_DIR = PROJECT_ROOT / "datasets" / "Thai"
PROV_DS = THAI_DIR / "thai-car-license-plate-province.v5i.yolov11"
CHARBOX_DS = THAI_DIR / "LPR 2 - Character Box Detection.yolov11"
CHAR_META_CSV = THAI_DIR / "thai_character_crops" / "metadata.csv"
ABBR_MAP_PATH = cfg.WEIGHTS_DIR / "province_abbr_map.json"
PROV_MAP_PATH = cfg.WEIGHTS_DIR / "province_map.json"

OUT_DIR = THAI_DIR / "recognition_v2"
PLATE_DIR = OUT_DIR / "plate_crops_v2" / "all"
PROV_DIR = OUT_DIR / "province_crops_v2" / "all"
CHAR_DIR = OUT_DIR / "char_crops_v2" / "all"
MANIFEST_PATH = OUT_DIR / "manifest.csv"
REJECT_PATH = OUT_DIR / "rejected.csv"

SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# Model loading (mirrors LPRPipelineService.__init__ so extraction == serve)
# ---------------------------------------------------------------------------
def load_models(device: str):
    m2_path = cfg.ACTIVE_MODEL_2_PATH
    m3a_path = cfg.ACTIVE_CHAR_BOX_MODEL_PATH

    m2_spec = apply_override(get_m2_spec(m2_path.name), cfg, "M2")
    m3a_spec = apply_override(get_m3a_spec(m3a_path.name), cfg, "M3A")

    def _load(path: Path, spec, names):
        low = path.name.lower()
        if "rfdetr" in low and path.exists():
            raise RuntimeError(
                "RF-DETR M2/M3A extraction is not wired here — rfdetr.predict() "
                "needs the live-wrapper path. Point MODEL_2_FILENAME / "
                "MODEL_3A_FILENAME at a LibreYOLO or Ultralytics checkpoint."
            )
        if path.name.endswith("_rtdetr.pt"):
            return RTDETR(str(path))
        if any(k in low for k in ("dfine", "libre", "picodet", "rtdetrv2", "obb")):
            from libreyolo import LibreYOLO
            tensor = spec.tensor_w if spec.tensor_w == spec.tensor_h else None
            return LibreYOLOWrapper(LibreYOLO(str(path)), names=names,
                                    device=device, default_imgsz=tensor)
        from ultralytics import YOLO
        return YOLO(str(path))

    print(f"Loading M2  : {m2_path.name}  (spec {m2_spec.tensor_w}x{m2_spec.tensor_h}, "
          f"train aspect {m2_spec.train_aspect_w_over_h})")
    print(f"Loading M3A : {m3a_path.name}  (spec {m3a_spec.tensor_w}x{m3a_spec.tensor_h}, "
          f"train aspect {m3a_spec.train_aspect_w_over_h})")
    model_m2 = _load(m2_path, m2_spec, ["plate_char", "province"])
    model_m3a = _load(m3a_path, m3a_spec, ["char"])
    return model_m2, m2_spec, model_m3a, m3a_spec


def predict(model, img, conf: float):
    """Single-image predict that works for YOLO, RTDETR and LibreYOLO wrappers."""
    try:
        return model(img, conf=conf, verbose=False)[0]
    except TypeError:
        return model(img, conf=conf)[0]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def load_province_maps():
    with open(ABBR_MAP_PATH, "r", encoding="utf-8") as f:
        abbr_map = json.load(f)          # ABBR -> {province_id, thai_name}
    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)          # id(str) -> thai_name
    return abbr_map, prov_map


def load_prov_class_names():
    """data.yaml `names` list -> index -> ABBR."""
    with open(PROV_DS / "data.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names") or []
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}


def load_char_gt() -> dict[str, str]:
    """source_image -> GT plate text (the surviving copy of gt_plate_char.csv)."""
    gt: dict[str, str] = {}
    if not CHAR_META_CSV.exists():
        return gt
    with open(CHAR_META_CSV, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src = (row.get("source_image") or "").strip()
            raw = (row.get("raw_gt_plate") or "").strip()
            if src and raw:
                gt[src] = raw
    return gt


def clean_gt_chars(raw: str) -> list[str]:
    """GT text -> ordered list of characters (drops spaces / dashes)."""
    return [c for c in raw if c not in " -–—_·."]


def read_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    """Read a YOLO txt into [(cls, x1, y1, x2, y2)] in pixels."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = (float(v) for v in parts[1:5])
        x1 = int(round((xc - bw / 2) * img_w))
        y1 = int(round((yc - bh / 2) * img_h))
        x2 = int(round((xc + bw / 2) * img_w))
        y2 = int(round((yc + bh / 2) * img_h))
        out.append((cls, max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)))
    return out


# ---------------------------------------------------------------------------
# Cropping helpers — formulas must stay identical to api_server.py
# ---------------------------------------------------------------------------
def pad_to_square(crop_bgr: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Identical to extract_thai_character_crops.make_square_padded / serve pad."""
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    max_dim = max(h, w)
    corners = np.array([crop_bgr[0, 0], crop_bgr[0, -1],
                        crop_bgr[-1, 0], crop_bgr[-1, -1]])
    bg_color = np.median(corners, axis=0).astype(np.uint8)
    padded = np.full((max_dim, max_dim, 3), bg_color, dtype=np.uint8)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = crop_bgr
    return cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)


def m2_province_crop(plate: np.ndarray, box) -> np.ndarray | None:
    """Reproduces api_server.py:1796-1813 exactly."""
    rh, rw = plate.shape[:2]
    bx1, by1, bx2, by2 = box
    if not (by2 > int(rh * 0.50) and (by1 + by2) / 2 > int(rh * 0.45)):
        return None
    bw, bh_box = bx2 - bx1, by2 - by1
    pad_x = min(14, max(6, int(bw * 0.08)))
    pad_y = min(8, max(4, int(bh_box * 0.10)))
    px1 = max(0, bx1 - pad_x)
    px2 = min(rw, bx2 + pad_x)
    py1 = max(int(rh * 0.46), by1 - pad_y)
    py2 = min(rh, by2 + pad_y)
    if px2 <= px1 or py2 <= py1:
        return None
    crop = plate[py1:py2, px1:px2]
    return crop if crop.size else None


def m2_char_crop(plate: np.ndarray, box) -> np.ndarray | None:
    """Reproduces api_server.py:1778-1794 exactly."""
    rh, rw = plate.shape[:2]
    bx1, by1, bx2, by2 = box
    if by1 >= int(rh * 0.65):
        return None
    bw_char, bh_char = bx2 - bx1, by2 - by1
    pad_cx = min(8, max(2, int(bw_char * 0.03)))
    pad_cy = min(6, max(2, int(bh_char * 0.04)))
    cx1 = max(int(rw * 0.03), bx1 - pad_cx)
    cx2 = min(int(rw * 0.97), bx2 + pad_cx)
    cy1 = max(int(rh * 0.02), by1 - pad_cy)
    cy2 = min(int(rh * 0.68), by2 + pad_cy)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = plate[cy1:cy2, cx1:cx2]
    return crop if crop.size else None


def m3a_char_boxes(char_row: np.ndarray, model_m3a, spec, conf: float):
    """Reproduces api_server.py:1916-1949 (upscale -> detect -> NMS -> split).

    Returns [(x1, y1, x2, y2, conf)] in char_row pixel coordinates.
    """
    det_input, scale = upscale_to_train_geometry(char_row, spec)
    res = predict(model_m3a, det_input, conf)

    raw = []
    for b in res.boxes:
        x1, y1, x2, y2 = [int(v) for v in divide_boxes_by_scale(
            np.asarray(b.xyxy[0].cpu().numpy()).reshape(1, 4), scale)[0]]
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 10:
            continue
        raw.append((x1, y1, x2, y2, float(b.conf[0])))

    raw = charbox_greedy_nms(raw, iou_thr=0.35)
    raw = split_wide_char_boxes(raw, char_row)
    raw.sort(key=lambda t: t[0])
    return raw


def save_jpg(path: Path, img: np.ndarray, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def scene_key(p: Path) -> str:
    """Group id for the leak-free splitter: source basename stem with Roboflow's
    random suffix removed (keeps <tag>_<frame>_<origin>)."""
    return p.name.split(".rf.")[0]


# ---------------------------------------------------------------------------
# PASS A — province GT dataset: rectified plate crops -> M2 -> prov + char row
# ---------------------------------------------------------------------------
def extract_province_pass(model_m2, m2_spec, abbr_map, prov_map, cls_names,
                          conf_m2: float, limit: int | None,
                          manifest: list, rejected: list):
    print("\n" + "=" * 78)
    print("PASS A — province GT (thai-car-license-plate-province.v5i.yolov11)")
    print("=" * 78)

    items = []
    for split in SPLITS:
        img_dir = PROV_DS / split / "images"
        if img_dir.exists():
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    items.append((split, p))
    if limit:
        items = items[:limit]
    print(f"  plate crops with province GT : {len(items)}")

    per_class = Counter()
    for split, img_path in tqdm(items, desc="  M2 on province plates"):
        key = scene_key(img_path)
        plate = cv2.imread(str(img_path))
        if plate is None:
            rejected.append({"pass": "province", "key": key, "reason": "unreadable"})
            continue

        lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        boxes = read_yolo_boxes(lbl, plate.shape[1], plate.shape[0])
        if not boxes:
            rejected.append({"pass": "province", "key": key, "reason": "no_yolo_label"})
            continue

        abbr = cls_names.get(boxes[0][0], "")
        info = abbr_map.get(abbr)
        if not info:
            rejected.append({"pass": "province", "key": key,
                             "reason": f"unknown_province_class:{abbr}"})
            continue
        pid, thai_name = int(info["province_id"]), info["thai_name"]
        folder = f"{pid:02d}_{thai_name}"

        # --- scene store (the M2 input domain, for future M2 retrains)
        save_jpg(PLATE_DIR / folder / f"{key}.jpg", plate)

        # --- M2 in TRAIN pixel geometry, identical to api_server.py:1756
        m2_input, m2_scale = upscale_to_train_geometry(plate, m2_spec)
        res2 = predict(model_m2, m2_input, conf_m2)

        best_char = best_prov = None
        for c_box in res2.boxes:
            idx = int(c_box.cls[0])
            c_name = str(res2.names.get(idx, "")).lower()
            c_conf = float(c_box.conf[0])
            bx1, by1, bx2, by2 = [int(v) for v in divide_boxes_by_scale(
                np.asarray(c_box.xyxy[0].cpu().numpy()).reshape(1, 4), m2_scale)[0]]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(plate.shape[1], bx2), min(plate.shape[0], by2)
            if ("plate" in c_name or "char" in c_name):
                if best_char is None or c_conf > best_char[4]:
                    best_char = (bx1, by1, bx2, by2, c_conf)
            elif "prov" in c_name:
                if best_prov is None or c_conf > best_prov[4]:
                    best_prov = (bx1, by1, bx2, by2, c_conf)

        if best_prov is None:
            rejected.append({"pass": "province", "key": key, "reason": "m2_no_province_box",
                             "province": thai_name})
            continue

        prov_crop = m2_province_crop(plate, best_prov[:4])
        if prov_crop is None:
            rejected.append({"pass": "province", "key": key,
                             "reason": "m2_province_box_geometry_gate",
                             "province": thai_name})
            continue

        rel_prov = f"province_crops_v2/all/{folder}/{key}.jpg"
        save_jpg(OUT_DIR / rel_prov, prov_crop)
        manifest.append({
            "task": "province",
            "key": key,
            "source_split": split,
            "label": thai_name,
            "label_id": pid,
            "rel_path": rel_prov,
            "crop_w": prov_crop.shape[1],
            "crop_h": prov_crop.shape[0],
            "m2_prov_conf": round(best_prov[4], 4),
            "province_box_source": "m2",
        })
        per_class[thai_name] += 1

        # --- char row crop (M2 plate_char) stored for the char pass of this plate
        if best_char is not None:
            char_row = m2_char_crop(plate, best_char[:4])
            if char_row is not None:
                rel_row = f"char_row_crops_v2/all/{folder}/{key}.jpg"
                save_jpg(OUT_DIR / rel_row, char_row)
                manifest.append({
                    "task": "char_row",
                    "key": key,
                    "source_split": split,
                    "label": thai_name,
                    "label_id": pid,
                    "rel_path": rel_row,
                    "crop_w": char_row.shape[1],
                    "crop_h": char_row.shape[0],
                    "m2_prov_conf": round(best_char[4], 4),
                    "province_box_source": "m2",
                })

    print(f"  accepted province crops : {len(per_class)} classes / "
          f"{sum(per_class.values())} crops")
    print(f"  min per province        : {min(per_class.values()) if per_class else 0}")


# ---------------------------------------------------------------------------
# PASS B — character GT: char-row crops -> M3A -> per-character crops
#
# THE COUNT FILTER (critical, per the plan's warning): M3A boxes are only
# accepted when the number of boxes EXACTLY equals the number of characters in
# the GT plate text. Otherwise the box->character zip would silently attach the
# wrong label and we would retrain on the same mistakes.
# ---------------------------------------------------------------------------
def extract_char_pass(model_m3a, m3a_spec, char_gt: dict[str, str],
                      conf_m3a: float, limit: int | None,
                      manifest: list, rejected: list):
    print("\n" + "=" * 78)
    print("PASS B — character GT (LPR 2 - Character Box Detection.yolov11)")
    print("=" * 78)
    print(f"  GT text available for {len(char_gt)} source crops")

    items = []
    for split in SPLITS:
        img_dir = CHARBOX_DS / split / "images"
        if img_dir.exists():
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    items.append((split, p))
    if limit:
        items = items[:limit]
    print(f"  char-row crops : {len(items)}")

    per_char = Counter()
    n_mismatch = n_ok = 0

    for split, img_path in tqdm(items, desc="  M3A on char rows"):
        key = scene_key(img_path)
        raw_gt = char_gt.get(img_path.name)
        if not raw_gt:
            rejected.append({"pass": "char", "key": key, "reason": "no_gt_text"})
            continue
        gt_chars = clean_gt_chars(raw_gt)
        if len(gt_chars) < 2:
            rejected.append({"pass": "char", "key": key, "reason": "gt_too_short",
                             "gt": raw_gt})
            continue

        char_row = cv2.imread(str(img_path))
        if char_row is None:
            rejected.append({"pass": "char", "key": key, "reason": "unreadable"})
            continue

        boxes = m3a_char_boxes(char_row, model_m3a, m3a_spec, conf_m3a)

        if len(boxes) != len(gt_chars):
            n_mismatch += 1
            rejected.append({"pass": "char", "key": key,
                             "reason": f"box_count_mismatch:{len(boxes)}!={len(gt_chars)}",
                             "gt": raw_gt})
            continue
        n_ok += 1

        for i, ((x1, y1, x2, y2, bconf), ch) in enumerate(zip(boxes, gt_chars)):
            patch = char_row[max(0, y1):min(char_row.shape[0], y2),
                             max(0, x1):min(char_row.shape[1], x2)]
            if patch.shape[0] < 4 or patch.shape[1] < 4:
                continue
            sq = pad_to_square(patch, target_size=64)
            rel = f"char_crops_v2/all/{ch}/{key}_b{i}.jpg"
            save_jpg(OUT_DIR / rel, sq)
            manifest.append({
                "task": "char",
                "key": key,
                "source_split": split,
                "label": ch,
                "label_id": -1,
                "rel_path": rel,
                "crop_w": sq.shape[1],
                "crop_h": sq.shape[0],
                "m2_prov_conf": "",
                "province_box_source": "m3a",
                "m3a_conf": round(bconf, 4),
                "char_index": i,
                "gt_text": raw_gt,
            })
            per_char[ch] += 1

    print(f"  count-matched plates : {n_ok}   rejected by count filter : {n_mismatch}")
    print(f"  char classes         : {len(per_char)} / crops {sum(per_char.values())}")
    if per_char:
        counts = sorted(per_char.values())
        print(f"  per-class min/median/max : {counts[0]} / {counts[len(counts) // 2]} / {counts[-1]}")
        rare = [f"{c}:{n}" for c, n in per_char.most_common()[-8:]]
        print(f"  rarest classes : {', '.join(rare)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MANIFEST_FIELDS = ["task", "key", "source_split", "label", "label_id", "rel_path",
                   "crop_w", "crop_h", "m2_prov_conf", "province_box_source",
                   "m3a_conf", "char_index", "gt_text"]


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({k for r in rows for k in r} - set(fields))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + extra)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(
        description="Extract v2 province/character crops with the live pipeline")
    ap.add_argument("--only", choices=["province", "char", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit images per pass (smoke test)")
    ap.add_argument("--conf-m2", type=float, default=0.25)
    ap.add_argument("--conf-m3a", type=float, default=0.10,
                    help="keep low (NMS + exact GT count filter do the quality control)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if OUT_DIR.exists() and args.overwrite:
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    abbr_map, prov_map = load_province_maps()
    cls_names = load_prov_class_names()
    char_gt = load_char_gt()

    device = str(cfg.DEVICE)
    t0 = time.time()
    model_m2, m2_spec, model_m3a, m3a_spec = load_models(device)

    manifest: list[dict] = []
    rejected: list[dict] = []

    if args.only in ("province", "all"):
        extract_province_pass(model_m2, m2_spec, abbr_map, prov_map, cls_names,
                              args.conf_m2, args.limit, manifest, rejected)
    if args.only in ("char", "all"):
        extract_char_pass(model_m3a, m3a_spec, char_gt, args.conf_m3a,
                          args.limit, manifest, rejected)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for task, n in sorted(Counter(r["task"] for r in manifest).items()):
        uniq = len({r["key"] for r in manifest if r["task"] == task})
        print(f"  {task:<9} crops={n:<6} from {uniq} source crops")

    _write_csv(MANIFEST_PATH, manifest, MANIFEST_FIELDS)
    _write_csv(REJECT_PATH, rejected, ["pass", "key", "reason", "gt", "province"])
    print(f"\n  elapsed: {time.time() - t0:.1f}s")
    print("  NEXT: python src/split_recognition_dataset_v2.py")


if __name__ == "__main__":
    main()




