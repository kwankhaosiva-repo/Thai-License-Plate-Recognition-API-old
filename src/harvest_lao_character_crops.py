"""
src/harvest_lao_character_crops.py

Harvests individual character crops (Lao consonants and digits) from Lao license plate
images using ground_truth_all.csv and the RT-DETR Character Box Detector.

Outputs organized training and validation sets:
  datasets/Lao/lao_character_crops/train/{character}/
  datasets/Lao/lao_character_crops/valid/{character}/
  weights/char_classifier_map_lao.json
"""

import os
import sys
import json
import re
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import RTDETR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg

CSV_PATH = PROJECT_ROOT / "datasets" / "Lao" / "lao-plate-dataset" / "ground_truth_all.csv"
IMG_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao-plate-dataset" / "images"
OUT_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_character_crops"
MAP_PATH = PROJECT_ROOT / "weights" / "char_classifier_map_lao.json"

VALID_LAO_CONSONANTS = set("ກຂຄງຈຊຍດຕຖທນບປຜຝພຟມຢຣລວສຫອຮ")
VALID_DIGITS = set("0123456789")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def extract_character_boxes_rtdetr(char_region: np.ndarray, box_model: RTDETR, expected_count: int = 6):
    """
    Extracts sorted character bounding boxes from the bottom character region of a Lao plate
    using RT-DETR Character Box Detector with horizontal NMS.
    """
    rh, rw = char_region.shape[:2]
    if rh < 15 or rw < 30:
        return []

    res = box_model(char_region, conf=0.15, verbose=False, device=DEVICE)[0]
    raw_boxes = []
    for b in res.boxes:
        bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
        bconf = float(b.conf[0])
        # Filter border artifacts
        if bx2 <= int(rw * 0.025) or bx1 >= int(rw * 0.975):
            continue
        raw_boxes.append([bx1, by1, bx2, by2, bconf])

    if not raw_boxes:
        return []

    # Horizontal NMS (suppress duplicate overlapping detections)
    raw_boxes.sort(key=lambda b: b[4], reverse=True)
    kept = []
    for b in raw_boxes:
        overlap = False
        for k in kept:
            inter_x = max(0, min(k[2], b[2]) - max(k[0], b[0]))
            min_w = min(k[2] - k[0], b[2] - b[0])
            if inter_x / float(max(min_w, 1)) > 0.40:
                overlap = True
                break
        if not overlap:
            kept.append(b)

    kept.sort(key=lambda b: b[0])

    # If 7 boxes kept and expected is 6, drop the one with lowest confidence
    if len(kept) == 7 and expected_count == 6:
        min_idx = int(np.argmin([b[4] for b in kept]))
        kept = [b for i, b in enumerate(kept) if i != min_idx]
        kept.sort(key=lambda b: b[0])

    if len(kept) == expected_count:
        return [b[:4] for b in kept]

    return []


def harvest_lao_characters(max_records: int = 0):
    print(f"\n=======================================================")
    print(f"--- Harvesting Lao Character Crops via RT-DETR Box ---")
    print(f"Device: {DEVICE}")
    print(f"Ground Truth CSV: {CSV_PATH}")
    print(f"=======================================================\n")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    box_model_path = cfg.ACTIVE_CHAR_BOX_MODEL_PATH
    print(f"Loading Character Box Detector from: {box_model_path}")
    box_model = RTDETR(str(box_model_path))

    # Parse valid records
    all_classes = set()
    records = []

    for idx, row in df.iterrows():
        fn = str(row["filename"]).strip()
        img_p = IMG_DIR / fn
        if not img_p.exists():
            continue

        raw_letters = str(row["letters"]).strip()
        digits_val = row["digits"]
        if pd.isna(digits_val):
            continue
        try:
            digits_str = f"{int(float(digits_val)):04d}"
        except Exception:
            continue

        letters = [c for c in raw_letters if c in VALID_LAO_CONSONANTS]
        digits = [c for c in digits_str if c in VALID_DIGITS]

        if len(letters) == 2 and len(digits) == 4:
            expected_chars = letters + digits
            for ch in expected_chars:
                all_classes.add(ch)
            split = "valid" if (idx % 8 == 0) else "train"
            records.append((img_p, expected_chars, split, fn))

        if max_records > 0 and len(records) >= max_records:
            break

    # Canonical 34 Lao classes (10 digits + 24 Lao consonants)
    sorted_classes = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "ກ", "ຂ", "ຄ", "ງ", "ຈ", "ຊ", "ຍ", "ດ", "ຕ", "ຖ", "ທ", "ນ", "ບ", "ປ", "ຜ", "ພ", "ມ", "ຣ", "ລ", "ວ", "ສ", "ຫ", "ອ", "ຮ"
    ]
    char_map = {str(i): c for i, c in enumerate(sorted_classes)}
    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(char_map, f, ensure_ascii=False, indent=2)

    print(f"Discovered {len(sorted_classes)} classes: {sorted_classes}")
    print(f"Total valid candidate plate images: {len(records)}")

    # Ensure output directories exist
    for split in ["train", "valid"]:
        for c in sorted_classes:
            (OUT_DIR / split / c).mkdir(parents=True, exist_ok=True)

    harvested_counts = {c: 0 for c in sorted_classes}
    success_plates = 0

    print("Harvesting individual character crops with RT-DETR...")
    for img_p, expected_chars, split, fn in tqdm(records):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Standardize plate height to 160px for consistent box detection
        scale = 160.0 / max(h, 1)
        resized = cv2.resize(img, (int(w * scale), 160), interpolation=cv2.INTER_CUBIC)
        char_region = resized[int(160 * 0.35):, :]

        boxes = extract_character_boxes_rtdetr(char_region, box_model, expected_count=6)

        if len(boxes) == 6:
            success_plates += 1
            stem = Path(fn).stem
            cr_h, cr_w = char_region.shape[:2]

            for i, (bx1, by1, bx2, by2) in enumerate(boxes):
                # Add 2px margin around character
                cx1 = max(0, bx1 - 2)
                cy1 = max(0, by1 - 2)
                cx2 = min(cr_w, bx2 + 2)
                cy2 = min(cr_h, by2 + 2)

                ch_crop = char_region[cy1:cy2, cx1:cx2]
                if ch_crop.size == 0 or ch_crop.shape[0] < 4 or ch_crop.shape[1] < 4:
                    continue

                # Square pad with border color
                ch_h, ch_w = ch_crop.shape[:2]
                smax = max(ch_h, ch_w)
                corners = np.array([ch_crop[0, 0], ch_crop[0, -1], ch_crop[-1, 0], ch_crop[-1, -1]])
                bg_col = np.median(corners, axis=0).astype(np.uint8)
                padded = np.full((smax, smax, 3), bg_col, dtype=np.uint8)
                y_off = (smax - ch_h) // 2
                x_off = (smax - ch_w) // 2
                padded[y_off:y_off + ch_h, x_off:x_off + ch_w] = ch_crop
                crop_64 = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_CUBIC)

                label = expected_chars[i]
                save_p = OUT_DIR / split / label / f"{stem}_c{i}.jpg"
                cv2.imwrite(str(save_p), crop_64)
                harvested_counts[label] += 1

    print("\n=======================================================")
    print(f"Harvesting Complete! Successfully harvested {success_plates} plates ({sum(harvested_counts.values())} character crops).")
    print(f"Class distribution:")
    for c, cnt in sorted(harvested_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  Class '{c}': {cnt} crops")
    print(f"Saved crops to: {OUT_DIR}")
    print("=======================================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=0, help="Max records to process (0 = all)")
    args = parser.parse_args()

    harvest_lao_characters(max_records=args.max_records)
