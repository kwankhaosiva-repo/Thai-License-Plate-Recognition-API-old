"""
src/balance_and_augment_lao_characters.py

Balances the Lao character training dataset by applying realistic photometric
and geometric augmentations (rotation angle, brightness/contrast, blur, morphology)
to minority classes, bringing all 30 active classes up to a balanced distribution.
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

TRAIN_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_character_crops" / "train"
VALID_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_character_crops" / "valid"
MAP_PATH = PROJECT_ROOT / "weights" / "char_classifier_map_lao.json"


def augment_image(img: np.ndarray) -> np.ndarray:
    """Applies a random combination of realistic augmentations to a 64x64 character crop."""
    h, w = img.shape[:2]
    out = img.copy()

    # Median corner color for clean border padding
    corners = np.array([out[0, 0], out[0, -1], out[-1, 0], out[-1, -1]])
    bg_color = [int(v) for v in np.median(corners, axis=0)]

    # 1. Angle rotation: ±3° to ±7°
    angle = random.uniform(-7.0, 7.0)
    scale = random.uniform(0.95, 1.05)
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)

    # 2. Lighting & Contrast adjustment
    alpha = random.uniform(0.75, 1.30)  # Contrast
    beta = random.uniform(-25.0, 25.0)  # Brightness
    out = np.clip(alpha * out.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 3. Random subtle blur (35% probability)
    if random.random() < 0.35:
        sigma = random.uniform(0.5, 1.0)
        out = cv2.GaussianBlur(out, (3, 3), sigma)

    # 4. Random morphology (25% probability: slight bold or slight thinning)
    if random.random() < 0.25:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.random() < 0.5:
            # Dilation (slightly thicker strokes)
            out = cv2.erode(out, kernel, iterations=1) if np.mean(bg_color) > 128 else cv2.dilate(out, kernel, iterations=1)
        else:
            # Erosion (slightly thinner strokes)
            out = cv2.dilate(out, kernel, iterations=1) if np.mean(bg_color) > 128 else cv2.erode(out, kernel, iterations=1)

    return out


def balance_and_augment(target_samples: int = 600, max_cap: int = 1200):
    print("=" * 70)
    print("⚖️ Balancing Lao Character Dataset with Realistic Augmentations")
    print(f"Train Directory : {TRAIN_DIR}")
    print(f"Valid Directory : {VALID_DIR}")
    print(f"Target Samples  : {target_samples} per class")
    print(f"Max Cap for 'ກ' : {max_cap}")
    print("=" * 70)

    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Class map not found: {MAP_PATH}")

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_char = json.load(f)

    classes_30 = list(idx_to_char.values())
    print(f"Active classes: {len(classes_30)}")

    # 1. Ensure Valid has at least 3-4 samples per class for fair evaluation
    print("\n[Step 1/3] Ensuring validation set has at least 3 samples per class...")
    for c in classes_30:
        v_dir = VALID_DIR / c
        v_dir.mkdir(parents=True, exist_ok=True)
        v_files = list(v_dir.glob("*.jpg"))

        if len(v_files) < 3:
            t_dir = TRAIN_DIR / c
            t_files = list(t_dir.glob("*.jpg"))
            needed = 3 - len(v_files)
            for i, f in enumerate(t_files[:needed]):
                dest = v_dir / f"val_transferred_{f.name}"
                if not dest.exists():
                    shutil.copy2(f, dest)
            print(f"  Class '{c}': Transferred {min(needed, len(t_files))} real samples to valid.")

    # 2. Balance Training Classes
    print(f"\n[Step 2/3] Augmenting minority classes up to {target_samples} samples...")
    total_added = 0

    for c in tqdm(classes_30, desc="Balancing Classes"):
        c_dir = TRAIN_DIR / c
        c_dir.mkdir(parents=True, exist_ok=True)

        # Get existing real crops (exclude previously generated augs if any)
        real_files = [f for f in c_dir.glob("*.jpg") if "_aug_" not in f.name]
        if not real_files:
            continue

        curr_count = len(list(c_dir.glob("*.jpg")))

        if curr_count < target_samples:
            needed = target_samples - curr_count
            # Generate augmentations
            aug_idx = 0
            while aug_idx < needed:
                src_file = random.choice(real_files)
                img = cv2.imread(str(src_file))
                if img is None:
                    continue

                aug_img = augment_image(img)
                aug_name = f"{src_file.stem}_aug_{aug_idx:04d}.jpg"
                cv2.imwrite(str(c_dir / aug_name), aug_img)
                aug_idx += 1
                total_added += 1

    print(f"\n--> Successfully generated {total_added} high-quality augmented images!")

    # 3. Summary of new distribution
    print("\n[Step 3/3] Final Class Distribution in Train:")
    final_counts = {}
    for c in classes_30:
        c_dir = TRAIN_DIR / c
        cnt = len(list(c_dir.glob("*.jpg"))) if c_dir.exists() else 0
        final_counts[c] = cnt

    for c, cnt in sorted(final_counts.items(), key=lambda x: -x[1]):
        print(f"  Class '{c}': {cnt} crops")

    total_train = sum(final_counts.values())
    print(f"\nTotal Training Samples: {total_train} (across {len(classes_30)} classes)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-samples", type=int, default=600)
    parser.add_argument("--max-cap", type=int, default=1200)
    args = parser.parse_args()

    balance_and_augment(target_samples=args.target_samples, max_cap=args.max_cap)
