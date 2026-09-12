"""
src/harvest_lao_character_crops_v2.py

High-Precision Lao Character Crop Harvester (Dashboard-Identical Pipeline).
Features:
1. Standardizes plate to 320x160.
2. Uses Dashboard's "Flip-and-Detect" Workflow (cv2.flip(0) + Model 2 Component Detector)
   to extract the full-height `char_crop` component with clean top and bottom margins.
3. Character Box Detection via Model 3A (rtdetr_char_box.pt) at conf=0.12.
4. Horizontal NMS + recover_character_boxes(is_lao=True) for faint/occluded characters.
5. Full-glyph extraction with safe padding + actual plate background color fill (no grey letter cuts).
6. Strict spatial zone validation (Box 0,1 = consonant zone, Box 2-5 = digit zone).
7. Descriptive filenames: `{plate_text}_{stem}_pos{i}_{char}.jpg`.
8. Automatically updates visual report at output/lao_character_verification/report.html.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from ultralytics import RTDETR, YOLO
from tqdm import tqdm

from src.config import cfg

CSV_PATH = PROJECT_ROOT / "datasets" / "Lao" / "lao-plate-dataset" / "ground_truth_all.csv"
IMG_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao-plate-dataset" / "images"
OUT_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_character_crops"

VALID_LAO_CONSONANTS = set("ກຂຄງຈຊຍດຕຖທນບປຜພມຣລວສຫອຮ")
VALID_DIGITS = set("0123456789")

DEVICE = "mps" if cv2.ocl.haveOpenCL() else "cpu"
try:
    import torch
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except ImportError:
    DEVICE = "cpu"


def recover_character_boxes(detected_boxes: List[Tuple], crop_w: int, crop_h: int) -> List[Tuple]:
    """Recovers missing character boxes from spatial gaps for Lao plates (2 letters + 4 digits)."""
    if not detected_boxes:
        return detected_boxes

    widths = [b[2] - b[0] for b in detected_boxes]
    med_w = int(np.median(widths)) if widths else 34
    med_h = int(np.median([b[3] - b[1] for b in detected_boxes])) if detected_boxes else int(crop_h * 0.85)
    mean_y1 = int(np.mean([b[1] for b in detected_boxes])) if detected_boxes else 2
    mean_y2 = int(np.mean([b[3] for b in detected_boxes])) if detected_boxes else crop_h - 2

    recovered = list(detected_boxes)

    # 1. Internal Gaps between adjacent boxes
    for i in range(len(detected_boxes) - 1):
        curr_b = detected_boxes[i]
        next_b = detected_boxes[i + 1]
        gap_w = next_b[0] - curr_b[2]
        gap_center = (curr_b[2] + next_b[0]) / 2.0

        # Normal gap between Lao 2 letters and 4 digits is ~1.2 to 2.2 * med_w
        if 0.28 * crop_w < gap_center < 0.45 * crop_w and gap_w < int(med_w * 2.3):
            continue

        if gap_w >= int(med_w * 1.45):
            missing_count = int(round(gap_w / max(med_w, 1)))
            slot_w = gap_w / (missing_count + 1)
            for m in range(1, missing_count + 1):
                nx1 = int(curr_b[2] + m * slot_w - med_w / 2)
                nx2 = nx1 + med_w
                nx1 = max(0, min(crop_w - 2, nx1))
                nx2 = max(nx1 + 8, min(crop_w, nx2))
                recovered.append((nx1, mean_y1, nx2, mean_y2, 0.50))

    recovered.sort(key=lambda b: b[0])

    # 2. Leading edge gap (missing first or second consonant)
    if len(recovered) < 6 and recovered:
        first_box = recovered[0]
        if first_box[0] > int(med_w * 1.1) and (first_box[0] / crop_w) > 0.08:
            nx2 = first_box[0] - 4
            nx1 = max(0, nx2 - med_w)
            if nx2 - nx1 >= 10:
                recovered.insert(0, (nx1, mean_y1, nx2, mean_y2, 0.45))

    # 3. Trailing edge gap (missing last digit)
    if len(recovered) < 6 and recovered:
        last_box = recovered[-1]
        space_right = crop_w - last_box[2]
        if space_right > int(med_w * 0.9):
            nx1 = last_box[2] + 4
            nx2 = min(crop_w, nx1 + med_w)
            if nx2 - nx1 >= 10:
                recovered.append((nx1, mean_y1, nx2, mean_y2, 0.45))

    recovered.sort(key=lambda b: b[0])
    return recovered


def extract_character_boxes_from_char_crop(char_crop: np.ndarray, char_box_model) -> List[Tuple]:
    """Runs Model 3A on the extracted char_crop, performing horizontal NMS and spatial gap recovery."""
    ch, cw = char_crop.shape[:2]
    try:
        try:
            res = char_box_model(char_crop, conf=0.12, verbose=False, device=DEVICE)[0]
        except Exception:
            res = char_box_model(char_crop, conf=0.12, verbose=False)[0]
    except Exception:
        return []

    raw_boxes = []
    for b in res.boxes:
        bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0]]
        bconf = float(b.conf[0])
        bw = bx2 - bx1
        bh = by2 - by1
        if bw < 8 or bh < 10:
            continue
        if bx1 <= 2 and bw < 12:
            continue
        if bx2 >= cw - 2 and bw < 12:
            continue
        raw_boxes.append((bx1, by1, bx2, by2, bconf))

    if not raw_boxes:
        return []

    # Horizontal NMS
    raw_boxes.sort(key=lambda x: x[4], reverse=True)
    kept_boxes = []
    for b in raw_boxes:
        bx1, by1, bx2, by2, bconf = b
        bw = bx2 - bx1
        overlap = False
        for kb in kept_boxes:
            kx1, ky1, kx2, ky2, _ = kb
            kw = kx2 - kx1
            inter_x = max(0, min(bx2, kx2) - max(bx1, kx1))
            min_w = min(bw, kw)
            if min_w > 0 and (inter_x / min_w) > 0.45:
                overlap = True
                break
        if not overlap:
            kept_boxes.append(b)

    sorted_boxes = sorted(kept_boxes, key=lambda item: item[0])
    recovered = recover_character_boxes(sorted_boxes, cw, ch)

    # If 7 boxes kept and expected is 6, drop lowest confidence box
    if len(recovered) == 7:
        min_idx = int(np.argmin([b[4] for b in recovered]))
        recovered = [b for i, b in enumerate(recovered) if i != min_idx]
        recovered.sort(key=lambda b: b[0])

    return recovered


def harvest_lao_characters_v2(max_records: int = 0, clean_only: bool = True):
    print("=" * 70)
    print("🚀 Starting Dashboard-Identical Lao Character Harvester")
    print(f"Device          : {DEVICE}")
    print(f"Ground Truth CSV: {CSV_PATH}")
    print(f"Target Directory: {OUT_DIR}")
    print(f"Clean Only      : {clean_only}")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if clean_only:
        df = df[df["reviewed"] == "yes"]

    print(f"Total candidate rows: {len(df)}")

    # Load Model 2 (Component Detector) & Model 3A (Char Box Detector)
    comp_model_path = cfg.ACTIVE_MODEL_2_PATH
    char_box_model_path = cfg.ACTIVE_CHAR_BOX_MODEL_PATH

    print(f"Loading Component Model from: {comp_model_path}")
    comp_model = None
    if comp_model_path.exists():
        comp_model = RTDETR(str(comp_model_path)) if "rtdetr" in str(comp_model_path).lower() else YOLO(str(comp_model_path))

    print(f"Loading Character Box Model from: {char_box_model_path}")
    char_box_model = RTDETR(str(char_box_model_path)) if "rtdetr" in str(char_box_model_path).lower() else YOLO(str(char_box_model_path))

    # 34 canonical Lao classes
    map_path = cfg.WEIGHTS_DIR / "char_classifier_map_lao.json"
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            classes_34 = list(json.load(f).values())
    else:
        classes_34 = [
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "ກ", "ຂ", "ຄ", "ງ", "ຈ", "ຊ", "ຍ", "ດ", "ຕ", "ຖ", "ທ", "ນ", "ບ", "ປ", "ຜ", "ພ", "ມ", "ຣ", "ລ", "ວ", "ສ", "ຫ", "ອ", "ຮ"
        ]

    for split in ["train", "valid"]:
        for c in classes_34:
            (OUT_DIR / split / c).mkdir(parents=True, exist_ok=True)

    harvested_counts = {c: 0 for c in classes_34}
    success_plates = 0
    skipped_spatial = 0

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
            split = "valid" if (idx % 8 == 0) else "train"
            records.append((img_p, expected_chars, split, fn, "".join(expected_chars)))

        if max_records > 0 and len(records) >= max_records:
            break

    print(f"Valid plates matching format (2 letters + 4 digits): {len(records)}")

    for img_p, expected_chars, split, fn, plate_text in tqdm(records, desc="Harvesting Crops"):
        orig_plate = cv2.imread(str(img_p))
        if orig_plate is None:
            continue

        # Standardize plate to 320x160 (Dashboard standard)
        rectified_plate = cv2.resize(orig_plate, (320, 160), interpolation=cv2.INTER_CUBIC)
        rh, rw = 160, 320

        # Dashboard Flip-and-Detect Workflow:
        # Flip plate vertically so characters are at the top, matching Thai Model 2 layout!
        flipped_plate = cv2.flip(rectified_plate, 0)
        char_crop = None

        if comp_model is not None:
            try:
                try:
                    res2 = comp_model(flipped_plate, conf=0.15, verbose=False, device=DEVICE)[0]
                except Exception:
                    res2 = comp_model(flipped_plate, conf=0.15, verbose=False)[0]

                best_char_conf = 0.0
                for c_box in res2.boxes:
                    c_idx = int(c_box.cls[0])
                    c_name = comp_model.names[c_idx].lower()
                    c_conf = float(c_box.conf[0])
                    bx1, by1, bx2, by2 = c_box.xyxy[0].cpu().numpy().astype(int)

                    # Map coordinates back to upright orientation: y_orig = rh - y_flipped
                    orig_y1 = max(0, rh - by2)
                    orig_y2 = min(rh, rh - by1)
                    orig_x1 = max(0, bx1)
                    orig_x2 = min(rw, bx2)

                    if ("plate" in c_name or "char" in c_name) and (c_conf > best_char_conf):
                        # Give a small 2px margin to ensure full letters are captured
                        pad_y1 = max(0, orig_y1 - 3)
                        pad_y2 = min(rh, orig_y2 + 3)
                        pad_x1 = max(0, orig_x1 - 2)
                        pad_x2 = min(rw, orig_x2 + 2)
                        comp_candidate = rectified_plate[pad_y1:pad_y2, pad_x1:pad_x2]
                        if comp_candidate.shape[0] >= 35 and comp_candidate.shape[1] >= 140:
                            char_crop = comp_candidate
                            best_char_conf = c_conf
            except Exception:
                pass

        # Fallback if Model 2 missed component
        if char_crop is None:
            char_y1, char_y2 = int(rh * 0.35), int(rh * 0.97)
            char_x1, char_x2 = int(rw * 0.04), int(rw * 0.96)
            char_crop = rectified_plate[char_y1:char_y2, char_x1:char_x2]

        ch_h, ch_w = char_crop.shape[:2]

        # Extract character boxes
        boxes = extract_character_boxes_from_char_crop(char_crop, char_box_model)

        if len(boxes) != 6:
            continue

        # Spatial Sanity Check:
        # Box 0,1 in consonant zone (cx/ch_w < 0.42)
        # Box 2 in digit zone (cx/ch_w > 0.22)
        cx0 = (boxes[0][0] + boxes[0][2]) / (2.0 * ch_w)
        cx1 = (boxes[1][0] + boxes[1][2]) / (2.0 * ch_w)
        cx2 = (boxes[2][0] + boxes[2][2]) / (2.0 * ch_w)

        if cx0 > 0.38 or cx1 > 0.45 or cx2 < 0.22:
            skipped_spatial += 1
            continue

        # Validated! Crop each character using Dashboard-exact glyph extraction
        success_plates += 1
        stem = Path(fn).stem

        for i, (bx1, by1, bx2, by2, _) in enumerate(boxes):
            # 3px margin to protect ascenders/descenders
            cx1 = max(0, bx1 - 3)
            cy1 = max(0, by1 - 3)
            cx2 = min(ch_w, bx2 + 3)
            cy2 = min(ch_h, by2 + 3)
            c_crop = char_crop[cy1:cy2, cx1:cx2]
            if c_crop.shape[0] < 6 or c_crop.shape[1] < 6:
                continue

            # Dashboard-exact Square Padding using Median Background Color
            sh, sw = c_crop.shape[:2]
            smax = max(sh, sw)
            corners = np.array([c_crop[0, 0], c_crop[0, -1], c_crop[-1, 0], c_crop[-1, -1]])
            bg_col = np.median(corners, axis=0).astype(np.uint8)

            padded = np.full((smax, smax, 3), bg_col, dtype=np.uint8)
            padded[(smax - sh) // 2 : (smax - sh) // 2 + sh, (smax - sw) // 2 : (smax - sw) // 2 + sw] = c_crop
            final_crop = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_CUBIC)

            target_char = expected_chars[i]
            out_name = f"{plate_text}_{stem}_pos{i}_{target_char}.jpg"
            out_path = OUT_DIR / split / target_char / out_name
            cv2.imwrite(str(out_path), final_crop)
            harvested_counts[target_char] += 1

    print("\n" + "=" * 70)
    print("🎉 Harvest Complete!")
    print(f"Successfully harvested {success_plates} plates ({sum(harvested_counts.values())} character crops).")
    print(f"Skipped misaligned plates (spatial filter): {skipped_spatial}")
    print("\nClass distribution:")
    for c, cnt in sorted(harvested_counts.items(), key=lambda x: -x[1]):
        print(f"  Class '{c}': {cnt} crops")
    print(f"Saved crops to: {OUT_DIR}")
    print("=" * 70)

    # Regenerate visual verification report
    print("\nUpdating visual verification report...")
    gen_script = PROJECT_ROOT / "src" / "verify_lao_character_crops.py"
    if gen_script.exists():
        import subprocess
        subprocess.run([sys.executable, str(gen_script)], check=False)
        print("Verification report updated at: output/lao_character_verification/report.html")


if __name__ == "__main__":
    harvest_lao_characters_v2(clean_only=True)
