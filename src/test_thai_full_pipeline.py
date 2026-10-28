#!/usr/bin/env python3
"""
src/test_thai_full_pipeline.py

Dedicated Local Full Pipeline Test Tool for Thai License Plate Recognition.
Evaluates end-to-end inference from raw car images through:
  - Model 1   : Plate Detection (PicoDet-S 320x320)
  - Warp      : 4-Corner Regressor (Perspective Homography 320x160)
  - Model 1.5 : Country Classifier (MobileNetV3-Small)
  - Model 2   : Component Detector (plate_char vs province, PicoDet-S 320x320)
  - Model 3A  : Character Box Detector (PicoDet-S 320x320)
  - Model 3B  : Character Classifier (MobileNetV2, 50 classes)
  - Model 3B/4: Province Classifier (ResNet18-Grayscale, 77 classes)

Outputs:
  - Formatted Thai License Plate Text
  - Classified Province & Confidence
  - Per-Stage Latency Breakdown (ms)
  - Visual Diagnostic Snapshots (annotated scene, rectified plate, character boxes, province banner)

Usage:
  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/test_thai_full_pipeline.py [image_or_folder] [--out-dir runs/test_thai_local]
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_server import LPRPipelineService
from src.config import cfg


def draw_detection_overlay(orig_bgr, res):
    """Draws bounding box, corners, text, and province onto the raw scene."""
    canvas = orig_bgr.copy()
    h, w = canvas.shape[:2]

    # Draw 4-corners if available
    corners = res.get("plate_corners")
    if corners:
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 240, 255), thickness=3)
        for pt in pts:
            cv2.circle(canvas, tuple(pt), 5, (0, 0, 255), -1)
            cv2.circle(canvas, tuple(pt), 2, (255, 255, 255), -1)
    else:
        # Fallback to plate_box
        box = res.get("plate_box")
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 240, 255), 3)

    # Text banner
    text = res.get("plate_text", "")
    prov = res.get("province", "")
    banner = f"{text} {prov}".strip()

    if banner:
        box = res.get("plate_box")
        tx = int(box[0]) if box else 20
        ty = max(30, int(box[1]) - 12) if box else 40
        # Shadow
        cv2.putText(canvas, banner, (tx + 2, ty + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3)
        cv2.putText(canvas, banner, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline Local Test for Thai License Plate Recognition")
    parser.add_argument("input_path", nargs="?", default="datasets/Thai/LPR_2_BBox/test/images",
                        help="Path to image file or directory of images")
    parser.add_argument("--out-dir", type=str, default="runs/test_thai_local",
                        help="Output directory to save visual diagnostic images")
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum number of images to test if input is a folder (default: 10)")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        img_files = [input_path]
    elif input_path.is_dir():
        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
            img_files.extend(input_path.glob(ext))
        img_files = sorted(img_files)[:args.limit]
    else:
        print(f"Error: Input path not found: {input_path}")
        return

    print("=" * 75)
    print("🇹🇭 Starting Local Full Pipeline Test on Thai License Plates")
    print("=" * 75)
    print(f"Input path : {input_path}")
    print(f"Images count: {len(img_files)}")
    print(f"Output dir  : {out_dir}")
    print("=" * 75)

    # Initialize Service (Loads all newly trained v6pico320 models)
    service = LPRPipelineService()

    total_images = len(img_files)
    success_count = 0
    latencies = []

    for idx, img_p in enumerate(img_files):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            print(f"[{idx+1}/{total_images}] Failed to read {img_p.name}")
            continue

        res = service.process_image(img_bgr, filename=str(img_p), debug=True)
        detected = res.get("detected", False)
        plate_text = res.get("plate_text", "")
        province = res.get("province", "")
        timing = res.get("timing", {})
        total_ms = timing.get("total_ms", 0)

        if detected:
            success_count += 1
            latencies.append(total_ms)

        print(f"\n[{idx+1}/{total_images}] {img_p.name}")
        print(f"  Detected   : {'✅ Yes' if detected else '❌ No'}")
        print(f"  Plate Text : {plate_text}")
        print(f"  Province   : {province}")
        print(f"  Confidence : Plate={res.get('confidence', {}).get('plate_detection')}, Prov={res.get('confidence', {}).get('province_classification')}")
        print(f"  Timing (ms): M1={timing.get('m1_ms')} | M2={timing.get('m2_ms')} | M3A_Box={timing.get('m3_box_det_ms')} | M3B_Char={timing.get('m3_char_cls_ms')} | M3B_Prov={timing.get('m3_prov_ms')} | Total={total_ms}ms")

        # Save Visual Snapshot
        overlay = draw_detection_overlay(img_bgr, res)
        save_path = out_dir / f"result_{img_p.stem}.jpg"
        cv2.imwrite(str(save_path), overlay)

    print("\n" + "=" * 75)
    print(f"📊 Summary: {success_count}/{total_images} plates recognized")
    if latencies:
        print(f"⚡ Average Latency: {int(np.mean(latencies))} ms per frame")
    print(f"🖼️ Diagnostic overlays saved to: {out_dir.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
