"""
src/test_interactive_lao_models.py

Interactive visual inspector for newly trained Lao LPR models.
Allows inspecting images one by one using Spacebar or arrow keys.

Controls:
  [Space] / [D] / [Right Arrow] : Next image
  [A]     / [Left Arrow]        : Previous image
  [S]                           : Save current annotated image
  [Q]     / [ESC]               : Exit

Usage:
  python src/test_interactive_lao_models.py
  python src/test_interactive_lao_models.py --input chacrit/25691019235909_character_detection/val
  python src/test_interactive_lao_models.py --input chacrit/province_more
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from libreyolo import LibreYOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DET_WEIGHTS = PROJECT_ROOT / "chacrit" / "output model" / "character_box_detector_picodet_s_lao_m3a.pt"
DEFAULT_CLS_WEIGHTS = PROJECT_ROOT / "chacrit" / "output model" / "character_classifier_lao_lao_char_v4.pth"
CHAR_MAP_PATH = PROJECT_ROOT / "weights" / "char_classifier_map_lao.json"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map_lao.json"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "chacrit" / "25691019235909_character_detection" / "val"

# System font with Lao Unicode glyph support
LAO_FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Lao Sangam MN.ttf",
]
FONT_PATH = next((p for p in LAO_FONT_PATHS if os.path.exists(p)), None)


def put_unicode_text(cv_img, text, pos, font_size=18, color=(255, 255, 255)):
    """Draw Unicode (including Lao/Thai script) on OpenCV BGR image."""
    x, y = pos
    if not FONT_PATH:
        cv2.putText(cv_img, text, (x, y + font_size), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return cv_img
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_PATH, font_size)
    # color is BGR in opencv, convert to RGB for PIL
    rgb_color = (color[2], color[1], color[0]) if len(color) == 3 else color
    draw.text((x, y), text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_char_classifier(weights_path: Path, char_map_path: Path, device: str = "cpu"):
    with open(char_map_path, "r", encoding="utf-8") as f:
        char_map = json.load(f)
    idx_to_char = {int(k): v for k, v in char_map.items()}

    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, len(idx_to_char))
    )
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, idx_to_char, tf


def draw_hud(img, filename, idx, total, text, latency_ms):
    h, w = img.shape[:2]
    # Header bar
    hud_h = 60
    hud = np.zeros((hud_h, w, 3), dtype=np.uint8)
    hud[:] = (30, 30, 30)

    # Info text
    title = f"[{idx+1}/{total}] {filename} | {latency_ms:.1f}ms"
    cv2.putText(hud, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    pred_str = f"Plate Text: {text}"
    hud = put_unicode_text(hud, pred_str, (12, 30), font_size=20, color=(0, 255, 128))

    # Footer controls hint
    footer_h = 30
    footer = np.zeros((footer_h, w, 3), dtype=np.uint8)
    footer[:] = (20, 20, 20)
    hint = "Controls: [Space/D] Next | [A] Prev | [S] Save | [Q] Quit"
    cv2.putText(footer, hint, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    combined = np.vstack([hud, img, footer])
    return combined


def main():
    parser = argparse.ArgumentParser(description="Interactive Image-by-Image Lao Model Tester")
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_INPUT_DIR),
                        help="Folder containing plate images to test")
    parser.add_argument("--conf", "-c", type=float, default=0.20,
                        help="Confidence threshold for character detector (default: 0.20)")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        help="Device to run inference on: 'cpu' or 'mps'")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Input folder does not exist: {input_dir}")
        sys.exit(1)

    print("=" * 70)
    print("🇱🇦 Lao LPR Interactive Image-by-Image Inspector")
    print(f"📁 Input Directory : {input_dir}")
    print(f"📦 M3A Detector    : {DEFAULT_DET_WEIGHTS.name}")
    print(f"🧠 Char Classifier : {DEFAULT_CLS_WEIGHTS.name}")
    print(f"⚙️ Device          : {args.device} (Confidence: {args.conf})")
    print("=" * 70)

    # 1. Load Detector
    print("Loading Character Box Detector...")
    detector = LibreYOLO(str(DEFAULT_DET_WEIGHTS))

    # 2. Load Classifier
    print("Loading Character Classifier...")
    classifier, idx_to_char, tf_char = load_char_classifier(
        DEFAULT_CLS_WEIGHTS, CHAR_MAP_PATH, device=args.device
    )

    # 3. Gather Images
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    img_files = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts])

    if not img_files:
        print(f"❌ No image files found in: {input_dir}")
        sys.exit(1)

    print(f"Found {len(img_files)} images to inspect.")
    print("\n👉 Window opened. Press [SPACE] to advance, [A] to go back, [Q] to quit.\n")

    current_idx = 0
    window_name = "Lao LPR Inspector (Press Space for Next)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while 0 <= current_idx < len(img_files):
        img_p = img_files[current_idx]
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            current_idx += 1
            continue

        h, w = img_bgr.shape[:2]
        t0 = time.time()

        # Step 1: Detect Character Bounding Boxes
        res = detector(img_bgr, conf=args.conf)
        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                bconf = float(b.conf[0])
                boxes.append((bx1, by1, bx2, by2, bconf))

        # Sort boxes left-to-right
        boxes = sorted(boxes, key=lambda x: x[0])

        # Step 2: Classify each detected character crop
        pred_chars = []
        vis_img = img_bgr.copy()

        for bx1, by1, bx2, by2, bconf in boxes:
            cx1 = max(0, bx1)
            cy1 = max(0, by1)
            cx2 = min(w, bx2)
            cy2 = min(h, by2)
            crop = img_bgr[cy1:cy2, cx1:cx2]

            char_text = "?"
            char_prob = 0.0

            if crop.size > 0 and crop.shape[0] >= 5 and crop.shape[1] >= 5:
                pil_c = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                ts = tf_char(pil_c).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    logits = classifier(ts)
                    probs = torch.softmax(logits, dim=1)
                    pidx = int(logits.argmax(dim=1).item())
                    char_prob = float(probs[0, pidx].item())
                    char_text = idx_to_char.get(pidx, "?")

            pred_chars.append(char_text)

            # Draw character bounding box
            cv2.rectangle(vis_img, (cx1, cy1), (cx2, cy2), (0, 220, 255), 2)

            # Draw character label tag
            tag = f"{char_text} {char_prob:.2f}"
            tag_h = 18
            tag_w = max(40, len(tag) * 10)
            cv2.rectangle(vis_img, (cx1, max(0, cy1 - tag_h - 2)), (cx1 + tag_w, cy1), (0, 220, 255), -1)
            vis_img = put_unicode_text(vis_img, tag, (cx1 + 2, max(0, cy1 - tag_h - 2)), font_size=13, color=(0, 0, 0))

        latency_ms = (time.time() - t0) * 1000.0
        plate_text = "".join(pred_chars) if pred_chars else "(No characters detected)"

        # Render display frame
        display_frame = draw_hud(vis_img, img_p.name, current_idx, len(img_files), plate_text, latency_ms)
        cv2.imshow(window_name, display_frame)

        # Wait for keypress
        key = cv2.waitKey(0) & 0xFF

        # Navigation handling
        if key in (32, ord('d'), ord('D'), 83):  # Space, D, Right Arrow
            current_idx += 1
        elif key in (ord('a'), ord('A'), 81):   # A, Left Arrow
            current_idx = max(0, current_idx - 1)
        elif key in (ord('s'), ord('S')):       # Save snapshot
            out_snap = PROJECT_ROOT / f"inspect_{img_p.stem}.jpg"
            cv2.imwrite(str(out_snap), display_frame)
            print(f"📸 Saved snapshot to: {out_snap}")
        elif key in (27, ord('q'), ord('Q')):   # ESC or Q to quit
            break

    cv2.destroyAllWindows()
    print("Inspector closed.")


if __name__ == "__main__":
    main()
