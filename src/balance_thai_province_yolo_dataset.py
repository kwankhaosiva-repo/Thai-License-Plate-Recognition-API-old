"""
src/balance_thai_province_yolo_dataset.py

Balances the 77-class Thai license plate province YOLO dataset
(datasets/Thai/thai-car-license-plate-province.v5i.yolov11) by:
1. Copying existing validation and test sets as-is to preserve benchmark integrity.
2. For the training set:
   - Capping dominant classes (e.g., BKK, CBI, SPK) to MAX_SAMPLES (150).
   - Augmenting minority classes (< MIN_TARGET_SAMPLES, e.g. 60) using realistic photometric
     and geometric transforms (brightness, contrast, yellow truck tinting, noise, blur, affine).
3. Writing the balanced dataset to datasets/Thai/thai_province_yolo_balanced/ with an updated data.yaml.
"""

import os
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DATASET_DIR = PROJECT_ROOT / "datasets" / "Thai" / "thai-car-license-plate-province.v5i.yolov11"
DST_DATASET_DIR = PROJECT_ROOT / "datasets" / "Thai" / "thai_province_yolo_balanced"

MIN_TARGET_SAMPLES = 60
MAX_SAMPLES = 150


def augment_image_and_boxes(img_bgr: np.ndarray, bboxes: list, aug_idx: int):
    """
    Applies realistic physical and photometric augmentations to plate image.
    bboxes is a list of [cls_id, x_c, y_c, w, h] in normalized coordinates [0, 1].
    """
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()
    new_boxes = [list(b) for b in bboxes]

    # Mode 1: Commercial Yellow Plate Tinting (~35% of augmentations)
    if aug_idx % 3 == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        yellow_tint = np.array([25, 200, 240], dtype=np.float32)  # BGR amber/yellow
        float_img = img.astype(np.float32)
        bg_pixels = (mask == 255)
        float_img[bg_pixels] = float_img[bg_pixels] * 0.45 + yellow_tint * 0.55
        img = np.clip(float_img, 0, 255).astype(np.uint8)

    # Mode 2: Lighting / Gamma / Exposure
    gamma = random.uniform(0.65, 1.45)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    img = cv2.LUT(img, table)

    # Mode 3: Contrast & Brightness jitter
    alpha = random.uniform(0.75, 1.30)
    beta = random.randint(-25, 25)
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Mode 4: Subtle blur or noise
    aug_r = random.random()
    if aug_r < 0.25:
        k_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k_size, k_size), 0)
    elif aug_r < 0.45:
        # Subtle sensor noise
        noise = np.random.normal(0, random.uniform(4, 12), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Mode 5: Slight Affine translation/tilt (~50% chance)
    if random.random() < 0.50:
        angle = random.uniform(-4.0, 4.0)
        scale = random.uniform(0.96, 1.04)
        tx = random.uniform(-0.02 * w, 0.02 * w)
        ty = random.uniform(-0.02 * h, 0.02 * h)

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Update normalized bounding box coordinates
        updated_boxes = []
        for cls_id, xc, yc, bw, bh in new_boxes:
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h

            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            pts_t = cv2.transform(pts.reshape(-1, 1, 2), M).reshape(-1, 2)

            nx1 = max(0.0, min(float(pts_t[:, 0].min()), w - 1.0))
            ny1 = max(0.0, min(float(pts_t[:, 1].min()), h - 1.0))
            nx2 = max(0.0, min(float(pts_t[:, 0].max()), w - 1.0))
            ny2 = max(0.0, min(float(pts_t[:, 1].max()), h - 1.0))

            n_bw = (nx2 - nx1) / w
            n_bh = (ny2 - ny1) / h
            n_xc = (nx1 + nx2) / (2.0 * w)
            n_yc = (ny1 + ny2) / (2.0 * h)

            if n_bw > 0.05 and n_bh > 0.05:
                updated_boxes.append([cls_id, n_xc, n_yc, n_bw, n_bh])
        new_boxes = updated_boxes

    return img, new_boxes


def balance_dataset():
    print("=" * 70)
    print("⚖️  Balancing Thai Province YOLO Dataset (77 Classes)")
    print(f"📁 Source: {SRC_DATASET_DIR}")
    print(f"🎯 Target: {DST_DATASET_DIR}")
    print(f"🎯 Policy: Target Min = {MIN_TARGET_SAMPLES}, Max Cap = {MAX_SAMPLES}")
    print("=" * 70)

    if not SRC_DATASET_DIR.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_DATASET_DIR}")

    # Load original data.yaml
    src_yaml_path = SRC_DATASET_DIR / "data.yaml"
    with open(src_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg["names"]
    num_classes = len(names)
    print(f"Loaded {num_classes} province classes from data.yaml.")

    # Prepare destination directories
    if DST_DATASET_DIR.exists():
        shutil.rmtree(DST_DATASET_DIR)
    for split in ["train", "valid", "test"]:
        (DST_DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # 1. Copy valid and test sets exactly as-is
    for split in ["valid", "test"]:
        src_split_imgs = SRC_DATASET_DIR / split / "images"
        src_split_lbls = SRC_DATASET_DIR / split / "labels"
        dst_split_imgs = DST_DATASET_DIR / split / "images"
        dst_split_lbls = DST_DATASET_DIR / split / "labels"

        if src_split_imgs.exists():
            img_files = list(src_split_imgs.glob("*.*"))
            print(f"Copying {len(img_files)} original {split} samples...")
            for img_p in img_files:
                lbl_p = src_split_lbls / f"{img_p.stem}.txt"
                shutil.copy2(img_p, dst_split_imgs / img_p.name)
                if lbl_p.exists():
                    shutil.copy2(lbl_p, dst_split_lbls / lbl_p.name)

    # 2. Analyze train split by class
    src_train_imgs = SRC_DATASET_DIR / "train" / "images"
    src_train_lbls = SRC_DATASET_DIR / "train" / "labels"
    dst_train_imgs = DST_DATASET_DIR / "train" / "images"
    dst_train_lbls = DST_DATASET_DIR / "train" / "labels"

    class_samples = defaultdict(list)
    for img_p in src_train_imgs.glob("*.*"):
        lbl_p = src_train_lbls / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        boxes = []
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = [float(v) for v in parts[1:5]]
                    boxes.append([cls_id] + coords)

        if boxes:
            primary_cls = boxes[0][0]
            class_samples[primary_cls].append((img_p, boxes))

    print(f"\nAudit complete: Found samples across {len(class_samples)} / {num_classes} classes.")

    total_train_written = 0
    final_counts = defaultdict(int)

    # 3. Process each class: cap if > MAX_SAMPLES, augment if < MIN_TARGET_SAMPLES
    random.seed(42)
    np.random.seed(42)

    for cls_idx in tqdm(range(num_classes), desc="Balancing classes"):
        cls_name = names[cls_idx]
        samples = class_samples.get(cls_idx, [])
        n_orig = len(samples)

        if n_orig == 0:
            print(f"⚠️ Warning: Class {cls_idx} ({cls_name}) has 0 real samples in train split!")
            continue

        # Cap if too large
        if n_orig > MAX_SAMPLES:
            chosen_samples = random.sample(samples, MAX_SAMPLES)
        else:
            chosen_samples = list(samples)

        # Write selected original samples
        for img_p, boxes in chosen_samples:
            dst_name = f"{img_p.stem}_{cls_name}{img_p.suffix}"
            shutil.copy2(img_p, dst_train_imgs / dst_name)
            with open(dst_train_lbls / f"{Path(dst_name).stem}.txt", "w", encoding="utf-8") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
            total_train_written += 1
            final_counts[cls_idx] += 1

        # Augment if below MIN_TARGET_SAMPLES
        needed = MIN_TARGET_SAMPLES - len(chosen_samples)
        if needed > 0:
            aug_idx = 0
            while needed > 0:
                src_img_p, src_boxes = samples[aug_idx % len(samples)]
                img_mat = cv2.imread(str(src_img_p))
                if img_mat is None:
                    aug_idx += 1
                    continue

                aug_mat, aug_boxes = augment_image_and_boxes(img_mat, src_boxes, aug_idx)
                if not aug_boxes:
                    aug_idx += 1
                    continue

                aug_name = f"{src_img_p.stem}_aug{aug_idx}_{cls_name}.jpg"
                cv2.imwrite(str(dst_train_imgs / aug_name), aug_mat, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                with open(dst_train_lbls / f"{Path(aug_name).stem}.txt", "w", encoding="utf-8") as f:
                    for b in aug_boxes:
                        f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                total_train_written += 1
                final_counts[cls_idx] += 1
                needed -= 1
                aug_idx += 1

    # 4. Write new data.yaml with absolute/relative path
    balanced_yaml = {
        "path": str(DST_DATASET_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": num_classes,
        "names": names,
    }
    with open(DST_DATASET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(balanced_yaml, f, sort_keys=False, allow_unicode=True)

    print("\n" + "=" * 70)
    print("✅ Balanced Thai Province Dataset Created Successfully!")
    print(f"📊 Total Train Images: {total_train_written}")
    print(f"📊 Min Samples per Class: {min(final_counts.values())}")
    print(f"📊 Max Samples per Class: {max(final_counts.values())}")
    print(f"📁 Dataset YAML: {DST_DATASET_DIR / 'data.yaml'}")
    print("=" * 70)


if __name__ == "__main__":
    balance_dataset()
