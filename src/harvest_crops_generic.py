"""
src/harvest_crops_generic.py — generic crop harvester for v4 data expansion.

Harvests TRAIN-QUALITY crops from any folder of images (full car scenes OR
direct plate crops) using the CURRENT production pipeline (M1 -> M2 -> M3A +
province model), so harvested crops follow the same *function* as the serve
path — no train/serve mismatch.

WHAT IT PRODUCES (under --dest, default: datasets/Thai/harvest_v4/)
  province_crops/<NN_Name>/*.jpg   M2 serve-rule province crops (256x80 domain)
  char_rows/<key>.jpg              M2 serve-rule character-row crops
  char_crops/<char>/*.jpg          M3A per-character 64x64 crops (pseudo-label)
  ocr_crops/<key>.jpg              rectified plate crops (256x64 OCR domain)
  ocr_gt.csv                       OCR pseudo-text per crop (FOR HUMAN REVIEW)
  manifest.csv                     provenance: source, scene key, labels, confs
  rejected.csv                     every skipped source + exact reason

LABEL SOURCES (honesty first — pseudo-labels are NOT ground truth)
  province: self-training via the ACTIVE production Thai province model
            (default, --prov-label-mode model) with a confidence floor
            (--prov-conf, default 0.60), OR inherited from the source image's
            parent folder name when sources are organized as class folders
            (--prov-label-mode gt_folder).
  char:     OCR pseudo-text + M3A boxes zipped left-to-right; rows whose box
            count != char count are still saved as char_rows/ocr_crops but
            skipped for char_crops (no wrong labels).
  ocr:      model text written ONLY to ocr_gt.csv for review — never silently
            used as GT.

USAGE (smoke test first!)
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/harvest_crops_generic.py \
      --src "datasets/Thai/LPR 2 - Character Box Detection.yolov11/train/images" \
      --dest datasets/Thai/harvest_v4 --limit 30

  Full run:
  ... --src datasets/Thai/<any_image_folder> --dest datasets/Thai/harvest_v4

After harvest: review manifest/rejected, then run src/split_recognition_dataset_v2.py
(or the balancer) before any retrain.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

from src.config import cfg
from src.preprocess_registry import (
    apply_override,
    divide_boxes_by_scale,
    get_m1_spec,
    get_m2_spec,
    get_m3a_spec,
    upscale_to_train_geometry,
)
from src.api_server import (
    LibreYOLOWrapper,
    charbox_greedy_nms,
    filter_implausible_char_boxes,
    split_wide_char_boxes,
)
from src.models import ResNetCRNN, ResNetProvinceClassifier, best_path_decode
from src.preprocess import get_ocr_transforms, get_grayscale_prov_transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Model loading (mirrors LPRPipelineService.__init__ / extract_recognition_crops_v2)
# ---------------------------------------------------------------------------
def _load_yolo_style(path: Path, names, spec_getter, override_key):
    low = path.name.lower()
    if path.name.endswith("_rtdetr.pt"):
        from ultralytics import RTDETR
        return RTDETR(str(path))
    if any(k in low for k in ("dfine", "libre", "picodet", "rtdetrv2", "obb")):
        from libreyolo import LibreYOLO
        spec = apply_override(spec_getter(path.name), cfg, override_key)
        tensor = spec.tensor_w if spec.tensor_w == spec.tensor_h else None
        return LibreYOLOWrapper(LibreYOLO(str(path)), names=names, device="cpu",
                                default_imgsz=tensor)
    from ultralytics import YOLO
    return YOLO(str(path))


def load_detectors():
    m1_path = cfg.ACTIVE_MODEL_1_PATH
    m2_path = cfg.ACTIVE_MODEL_2_PATH
    m3a_path = cfg.ACTIVE_CHAR_BOX_MODEL_PATH
    model_m1 = _load_yolo_style(m1_path, ["plate"], get_m1_spec, "M1")
    model_m2 = _load_yolo_style(m2_path, ["plate_char", "province"], get_m2_spec, "M2")
    model_m3a = _load_yolo_style(m3a_path, ["char"], get_m3a_spec, "M3A")
    print(f"M1  : {m1_path.name}")
    print(f"M2  : {m2_path.name}")
    print(f"M3A : {m3a_path.name}")
    return model_m1, model_m2, model_m3a


def load_ocr():
    ocr_path = cfg.WEIGHTS_DIR / cfg.OCR_FILENAME
    with open(cfg.WEIGHTS_DIR / "int_to_char.json", "r", encoding="utf-8") as f:
        int_to_char = json.load(f)
    model = ResNetCRNN(1, len(int_to_char), hidden_size=256)
    ckpt = torch.load(ocr_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, int_to_char, get_ocr_transforms(False)


def load_char_cls_model():
    """Thai char classifier (v3) — used to cross-check OCR pseudo-labels
    (double-agreement) before a crop is saved into char_crops/<class>/."""
    from torchvision import models as _models
    path = cfg.WEIGHTS_DIR / cfg.CHAR_CLASSIFIER_THAI_FILENAME
    ckpt = torch.load(path, map_location="cpu")
    idx_to_char = ckpt.get("class_map")
    if idx_to_char is None:
        idx_to_char = json.loads((cfg.WEIGHTS_DIR / "char_classifier_map.json").read_text(encoding="utf-8"))
    n = len(idx_to_char)
    model = _models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(model.last_channel, n))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    map_inv = {int(k): v for k, v in idx_to_char.items()} if all(isinstance(k, int) for k in idx_to_char) \
        else {int(k): v for k, v in idx_to_char.items()}
    print(f"CharCls (double-check): {path.name} ({n} classes)")
    return model, map_inv, tf


def load_province_classifier():
    """Active production province model (v3 after the config switch) — used for
    self-training labels + confidence gating."""
    prov_path = cfg.ACTIVE_PROV_MODEL_THAI_PATH
    ckpt = torch.load(prov_path, map_location="cpu")
    backbone = ckpt.get("backbone", "resnet18")
    n_classes = int(ckpt.get("n_classes", 0))
    with open(cfg.WEIGHTS_DIR / "province_map.json", "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    if n_classes == 0:
        n_classes = len(prov_map)
    model = ResNetProvinceClassifier(n_classes=n_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tf = get_grayscale_prov_transforms(is_train=False)
    print(f"M3B (labeler): {prov_path.name} ({backbone}, {n_classes} classes)")
    return model, prov_map, tf


def predict(model, img, conf: float):
    try:
        return model(img, conf=conf, verbose=False)[0]
    except TypeError:
        return model(img, conf=conf)[0]


# ---------------------------------------------------------------------------
# Serve-identical crop rules (must stay identical to src/api_server.py)
# ---------------------------------------------------------------------------
def m2_province_crop(plate, box):
    rh, rw = plate.shape[:2]
    bx1, by1, bx2, by2 = box
    if not (by2 > int(rh * 0.50) and (by1 + by2) / 2 > int(rh * 0.45)):
        return None
    bw, bh_box = bx2 - bx1, by2 - by1
    pad_x = min(14, max(6, int(bw * 0.08)))
    pad_y = min(8, max(4, int(bh_box * 0.10)))
    px1, px2 = max(0, bx1 - pad_x), min(rw, bx2 + pad_x)
    py1, py2 = max(int(rh * 0.46), by1 - pad_y), min(rh, by2 + pad_y)
    if px2 <= px1 or py2 <= py1:
        return None
    crop = plate[py1:py2, px1:px2]
    return crop if crop.size else None


def m2_char_crop(plate, box):
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


def m3a_char_boxes(char_row, model_m3a, spec, conf):
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
    # v4 parity with serve: drop dash slivers / frame strips / sub-stroke
    # fragments so harvested char crops never contain junk geometry.
    raw = filter_implausible_char_boxes(raw, char_row, strict=True)
    raw.sort(key=lambda t: t[0])
    return raw


def pad_to_square(crop_bgr, target_size=64):
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), 255, dtype=np.uint8)
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    pil = ImageOps.autocontrast(pil, cutoff=2)   # serve contract (api_server char cls branch)
    crop_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    max_dim = max(h, w)
    corners = np.array([crop_bgr[0, 0], crop_bgr[0, -1],
                        crop_bgr[-1, 0], crop_bgr[-1, -1]])
    bg = np.median(corners, axis=0).astype(np.uint8)
    padded = np.full((max_dim, max_dim, 3), bg, dtype=np.uint8)
    y_off, x_off = (max_dim - h) // 2, (max_dim - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = crop_bgr
    return cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)


def save_jpg(path: Path, img, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def scene_key(p: Path) -> str:
    return p.stem.split(".rf.")[0]


def province_label_from_gt_folder(img_path: Path, prov_map: dict):
    """Source images organized as <NN_Name>/ or <name>/ class folders."""
    for parent in (img_path.parent, img_path.parent.parent):
        name = parent.name
        if f"{int(name.split('_')[0]):02d}_" in f"{name}_" and name.split("_")[0].isdigit():
            pid = int(name.split("_")[0])
            thai = prov_map.get(str(pid))
            if thai:
                return pid, thai
        # exact Thai name folder
        for pid, thai in prov_map.items():
            if name == thai:
                return int(pid), thai
    return None, None


# ---------------------------------------------------------------------------
# Main harvest loop
# ---------------------------------------------------------------------------
def harvest(args):
    src_dir = Path(args.src).resolve()
    dest = Path(args.dest).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"--src not found: {src_dir}")
    dest.mkdir(parents=True, exist_ok=True)

    model_m1, model_m2, model_m3a = load_detectors()
    ocr_model, int_to_char, tf_ocr = load_ocr()
    prov_model, prov_map, tf_prov = load_province_classifier()
    char_cls_model, char_cls_map_inv, tf_char_cls = load_char_cls_model()

    m2_spec = apply_override(get_m2_spec(cfg.ACTIVE_MODEL_2_PATH.name), cfg, "M2")
    m3a_spec = apply_override(get_m3a_spec(cfg.ACTIVE_CHAR_BOX_MODEL_PATH.name), cfg, "M3A")

    out_prov = dest / "province_crops"
    out_char = dest / "char_crops"
    out_rows = dest / "char_rows"
    out_ocr = dest / "ocr_crops"

    manifest: list[dict] = []
    rejected: list[dict] = []
    prov_counts: Counter = Counter()
    char_counts: Counter = Counter()
    n_ocr = n_rows = 0

    def reject(key, reason, **extra):
        rejected.append({"key": key, "reason": reason, **extra})

    # Gather source images (also supports <src>/{train,valid,test}/images layout)
    items: list[Path] = []
    for split in ("train", "valid", "test"):
        d = src_dir / split / "images"
        if d.is_dir():
            items.extend(sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS))
    if not items:
        if src_dir.is_file():
            items = [src_dir]
        else:
            items = sorted(p for p in src_dir.rglob("*")
                           if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.limit:
        items = items[: args.limit]
    print(f"Source images: {len(items)}  ->  {dest}")

    for img_path in tqdm(items, desc="harvest"):
        key = scene_key(img_path)
        img = cv2.imread(str(img_path))
        if img is None:
            reject(key, "unreadable")
            continue
        rh, rw = img.shape[:2]

        # --- M1 plate detection (skipped when the source IS a plate crop)
        plates = []
        looks_like_plate = args.assume_plates or (max(rh, rw) <= 400 and 1.2 <= rw / max(rh, 1) <= 4.0)
        if looks_like_plate:
            plates.append(img)
        else:
            m1_spec = apply_override(get_m1_spec(cfg.ACTIVE_MODEL_1_PATH.name), cfg, "M1")
            m1_input, m1_scale = upscale_to_train_geometry(img, m1_spec)
            res1 = predict(model_m1, m1_input, args.conf_m1)
            for b in res1.boxes:
                x1, y1, x2, y2 = [int(v) for v in divide_boxes_by_scale(
                    np.asarray(b.xyxy[0].cpu().numpy()).reshape(1, 4), m1_scale)[0]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(rw, x2), min(rh, y2)
                if x2 - x1 > 20 and y2 - y1 > 8:
                    plates.append(img[y1:y2, x1:x2])
        if not plates:
            reject(key, "no_plate_found")
            continue

        for pi, plate in enumerate(plates):
            pkey = key if len(plates) == 1 else f"{key}_p{pi}"
            ph, pw = plate.shape[:2]
            if ph < 24 or pw < 48:
                reject(pkey, "plate_too_small")
                continue

            # --- M2 component split (train geometry)
            m2_input, m2_scale = upscale_to_train_geometry(plate, m2_spec)
            res2 = predict(model_m2, m2_input, args.conf_m2)
            best_char = best_prov = None
            for c_box in res2.boxes:
                c_name = str(res2.names.get(int(c_box.cls[0]), "")).lower()
                c_conf = float(c_box.conf[0])
                bx1, by1, bx2, by2 = [int(v) for v in divide_boxes_by_scale(
                    np.asarray(c_box.xyxy[0].cpu().numpy()).reshape(1, 4), m2_scale)[0]]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(pw, bx2), min(ph, by2)
                if "prov" in c_name:
                    if best_prov is None or c_conf > best_prov[4]:
                        best_prov = (bx1, by1, bx2, by2, c_conf)
                elif "char" in c_name or "plate" in c_name:
                    if best_char is None or c_conf > best_char[4]:
                        best_char = (bx1, by1, bx2, by2, c_conf)

            # --- Province crop + label (model self-training or GT folder)
            if best_prov is not None:
                prov_crop = m2_province_crop(plate, best_prov[:4])
                if prov_crop is not None:
                    pid, thai_name, label_src = None, None, None
                    if args.prov_label_mode == "gt_folder":
                        pid, thai_name = province_label_from_gt_folder(img_path, prov_map)
                        label_src = "gt_folder"
                    if thai_name is None:
                        # self-training: active province model labels the crop
                        x = tf_prov(Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))).unsqueeze(0)
                        with torch.no_grad():
                            out = prov_model(x)
                        conf_p, cls_p = out.softmax(1).max(1)
                        if float(conf_p) >= args.prov_conf:
                            pid = int(cls_p)
                            thai_name = prov_map.get(str(pid))
                            label_src = f"m3b_pseudo_{float(conf_p):.2f}"
                    if thai_name:
                        fname = f"{pid:02d}_{thai_name}"
                        rel = f"{fname}/{pkey}.jpg"
                        save_jpg(out_prov / rel, prov_crop)
                        prov_counts[thai_name] += 1
                        manifest.append({
                            "task": "province", "key": pkey,
                            "source": str(img_path.relative_to(src_dir) if img_path.is_relative_to(src_dir) else img_path),
                            "label": thai_name, "label_id": pid,
                            "label_source": label_src,
                            "rel_path": f"province_crops/{rel}",
                            "crop_w": prov_crop.shape[1], "crop_h": prov_crop.shape[0],
                        })
                    else:
                        reject(pkey, f"province_label_low_conf_or_unknown")
            else:
                reject(pkey, "m2_no_province_box")

            # --- Char row + OCR pseudo-text
            ocr_text = ""
            char_row = None
            if best_char is not None:
                char_row = m2_char_crop(plate, best_char[:4])
            if char_row is not None and char_row.size:
                pil_row = Image.fromarray(cv2.cvtColor(char_row, cv2.COLOR_BGR2RGB))
                x = tf_ocr(pil_row).unsqueeze(0)
                with torch.no_grad():
                    out = ocr_model(x)
                ocr_text = best_path_decode(out.softmax(-1), int_to_char)[0]
                rel = f"{pkey}.jpg"
                save_jpg(out_rows / rel, char_row)
                save_jpg(out_ocr / rel, plate)
                n_rows += 1
                n_ocr += 1
                manifest.append({
                    "task": "char_row", "key": pkey, "source": str(img_path),
                    "label": "", "label_source": "n/a",
                    "rel_path": f"char_rows/{rel}", "ocr_pseudo": ocr_text,
                    "crop_w": char_row.shape[1], "crop_h": char_row.shape[0],
                })

                # --- M3A per-character crops — CLASSIFIED BY MODEL 3A (char
                # classifier), the SAME flow as serve: M3A detector finds each
                # char box -> char classifier v3 assigns the class -> crop is
                # saved into char_crops/<predicted_class>/. OCR is NOT used to
                # decide folders; its whole-row reading is only stored in
                # char_rows + ocr_gt_REVIEW_ME.csv for human cross-checking.
                boxes = m3a_char_boxes(char_row, model_m3a, m3a_spec, args.conf_m3a)
                if not boxes:
                    reject(pkey, "m3a_no_char_boxes")
                    continue

                for bi, (bx1, by1, bx2, by2, bconf) in enumerate(boxes):
                    crop = char_row[max(0, by1):min(char_row.shape[0], by2),
                                    max(0, bx1):min(char_row.shape[1], bx2)]
                    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
                        continue
                    crop64 = pad_to_square(crop)
                    pil_c = Image.fromarray(cv2.cvtColor(crop64, cv2.COLOR_BGR2RGB))
                    x_c = tf_char_cls(pil_c).unsqueeze(0)
                    with torch.no_grad():
                        probs = torch.softmax(char_cls_model(x_c), dim=1)[0]
                    cls_conf, cls_pred = probs.max(0)
                    cls_sym = char_cls_map_inv.get(int(cls_pred), "?")
                    if float(cls_conf) < args.char_conf:
                        reject(f"{pkey}_{bi}", "char_cls_low_conf",
                               char_cls=cls_sym, conf=round(float(cls_conf), 3))
                        continue
                    rel = f"{cls_sym}/{pkey}_{bi}.jpg"
                    save_jpg(out_char / rel, crop64)
                    char_counts[cls_sym] += 1
                    manifest.append({
                        "task": "char", "key": f"{pkey}_{bi}", "source": str(img_path),
                        "label": cls_sym, "label_source": f"m3a_cls_{float(cls_conf):.2f}",
                        "rel_path": f"char_crops/{rel}",
                        "crop_w": 64, "crop_h": 64, "m3a_conf": round(bconf, 4),
                        "ocr_pseudo": ocr_text,
                    })

    # --- Write reports
    with open(dest / "manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        if manifest:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in manifest for k in r}))
            w.writeheader()
            w.writerows(manifest)
    with open(dest / "rejected.csv", "w", encoding="utf-8-sig", newline="") as f:
        if rejected:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rejected for k in r}))
            w.writeheader()
            w.writerows(rejected)
    with open(dest / "ocr_gt_REVIEW_ME.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "ocr_pseudo_text"])
        for r in manifest:
            if r["task"] == "char_row":
                w.writerow([r["rel_path"].replace("char_rows/", "ocr_crops/"), r.get("ocr_pseudo", "")])

    print("\n" + "=" * 70)
    print(f"Province crops : {sum(prov_counts.values())} across {len(prov_counts)} classes")
    print(f"Char rows      : {n_rows}   (OCR pseudo-text -> ocr_gt_REVIEW_ME.csv)")
    print(f"OCR crops      : {n_ocr}")
    print(f"Char crops     : {sum(char_counts.values())} across {len(char_counts)} chars")
    n_mismatch = sum(1 for r in rejected if r.get("reason") == "char_pair_count_mismatch")
    if n_mismatch:
        print(f"                 ({n_mismatch} rows skipped: M3A box count != OCR char count — see rejected.csv)")
    print(f"Rejected       : {len(rejected)}  (see rejected.csv)")
    if char_counts:
        top = char_counts.most_common(5)
        low = sorted(char_counts.items(), key=lambda kv: kv[1])[:5]
        print(f"  most common : {top}")
        print(f"  rarest      : {low}")
    print("=" * 70)
    print("NEXT: review manifest.csv / ocr_gt_REVIEW_ME.csv, then balance + split")
    print("before retraining. Pseudo-labels are NOT ground truth.")


def main():
    p = argparse.ArgumentParser(description="Generic serve-identical crop harvester (v4 expansion)")
    p.add_argument("--src", required=True, help="Source folder of scene or plate images (or a single file)")
    p.add_argument("--dest", default="datasets/Thai/harvest_v4", help="Output root (default: datasets/Thai/harvest_v4)")
    p.add_argument("--limit", type=int, default=None, help="Limit source images (smoke test)")
    p.add_argument("--conf-m1", type=float, default=0.30)
    p.add_argument("--conf-m2", type=float, default=0.25)
    p.add_argument("--conf-m3a", type=float, default=0.10)
    p.add_argument("--assume-plates", action="store_true", help="Treat every source image as a plate crop (skip M1)")
    p.add_argument("--prov-label-mode", choices=["model", "gt_folder"], default="model",
                   help="model = self-training via active province model; gt_folder = inherit label from source parent folder")
    p.add_argument("--prov-conf", type=float, default=0.60,
                   help="Province self-label confidence floor (default 0.60)")
    p.add_argument("--char-conf", type=float, default=0.70,
                   help="Char classifier (Model 3A) confidence floor for char_crops (default 0.70)")
    p.add_argument("--no-char-crops", action="store_true",
                   help="DISABLE per-character crops (default: ON — pairs M3A boxes with OCR pseudo-text; count-mismatch rows are skipped)")
    args = p.parse_args()
    harvest(args)


if __name__ == "__main__":
    main()
