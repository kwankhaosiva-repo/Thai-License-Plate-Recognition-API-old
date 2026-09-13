"""
src/prepare_lao_province_yolo_dataset.py

Builds a balanced 18-class YOLO Object Detection dataset for Lao license plate provinces:
1. Reads ground truth from datasets/Lao/lao-plate-dataset/ground_truth_all.csv (11,317 plates).
2. Groups plates by province_id (0 to 17).
3. Splits into train, valid, and test sets.
4. Balances the train split:
   - Caps dominant classes (e.g. Vientiane 8,100 plates, Savannakhet 763 plates) to MAX_SAMPLES (150).
   - Augments minority classes (e.g. Phongsali 5 plates, Sekong 9 plates) up to TARGET_MIN_SAMPLES (120).
5. Exports to datasets/Lao/lao_province_yolo_balanced/ with data.yaml.
"""

import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAO_DATASET_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao-plate-dataset"
IMAGES_DIR = LAO_DATASET_DIR / "images"
CSV_PATH = LAO_DATASET_DIR / "ground_truth_all.csv"
MAP_PATH = PROJECT_ROOT / "weights" / "province_map_lao.json"

DST_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_yolo_balanced"

TARGET_MIN_SAMPLES = 120
MAX_SAMPLES = 150

# Standard empirical Lao province banner bounding box on rectified plates
# [xc, yc, w, h] in normalized coordinates
DEFAULT_PROV_BOX = [0.496, 0.209, 0.761, 0.321]


def augment_plate_and_box(img_bgr: np.ndarray, bbox: list, aug_idx: int):
    """
    Applies realistic photographic, lighting, and geometric augmentations to Lao plate image.
    bbox is [cls_id, xc, yc, w, h].
    """
    h, w = img_bgr.shape[:2]
    img = img_bgr.copy()
    cls_id, xc, yc, bw, bh = bbox

    # 1. Lighting / Gamma / Exposure
    gamma = random.uniform(0.70, 1.40)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    img = cv2.LUT(img, table)

    # 2. Contrast & Brightness jitter
    alpha = random.uniform(0.80, 1.25)
    beta = random.randint(-20, 20)
    img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 3. Subtle blur or noise
    if random.random() < 0.30:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif random.random() < 0.50:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Subtle Affine tilt/scale (~45% chance)
    if random.random() < 0.45:
        angle = random.uniform(-3.5, 3.5)
        scale = random.uniform(0.97, 1.03)
        tx = random.uniform(-0.015 * w, 0.015 * w)
        ty = random.uniform(-0.015 * h, 0.015 * h)

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # Transform box
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

        bw = (nx2 - nx1) / w
        bh = (ny2 - ny1) / h
        xc = (nx1 + nx2) / (2.0 * w)
        yc = (ny1 + ny2) / (2.0 * h)

    return img, [cls_id, xc, yc, bw, bh]


def prepare_dataset():
    print("=" * 70)
    print("🇱🇦 Preparing Balanced Lao Province YOLO Dataset (18 Classes)")
    print(f"📁 Source: {LAO_DATASET_DIR}")
    print(f"🎯 Target: {DST_DIR}")
    print("=" * 70)

    if not CSV_PATH.exists() or not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Lao dataset not found at: {LAO_DATASET_DIR}")

    import json
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)

    class_names = [prov_map[str(i)] for i in range(18)]
    num_classes = len(class_names)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from ground_truth_all.csv.")

    # Filter to existing images
    valid_records = []
    for _, row in df.iterrows():
        fname = str(row["filename"])
        img_p = IMAGES_DIR / fname
        if img_p.exists():
            pid = int(row["province_id"])
            if 0 <= pid < 18:
                valid_records.append((img_p, pid))

    print(f"Verified {len(valid_records)} images existing on disk.")

    # Group by province_id
    by_prov = defaultdict(list)
    for img_p, pid in valid_records:
        by_prov[pid].append(img_p)

    # Prepare destination directories
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    for split in ["train", "valid", "test"]:
        (DST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    total_counts = {"train": 0, "valid": 0, "test": 0}

    for pid in tqdm(range(num_classes), desc="Processing Lao provinces"):
        plist = by_prov.get(pid, [])
        if not plist:
            print(f"⚠️ Warning: Province ID {pid} ({class_names[pid]}) has 0 samples!")
            continue

        random.shuffle(plist)

        # Split: up to 15 for valid, 15 for test, remainder for train
        if len(plist) >= 50:
            n_val = min(20, int(len(plist) * 0.10))
            n_test = min(20, int(len(plist) * 0.10))
            test_imgs = plist[:n_test]
            val_imgs = plist[n_test : n_test + n_val]
            train_imgs = plist[n_test + n_val :]
        elif len(plist) >= 15:
            test_imgs = plist[:2]
            val_imgs = plist[2:4]
            train_imgs = plist[4:]
        else:
            # Very rare classes (e.g. 5–10 samples)
            test_imgs = plist[:1]
            val_imgs = plist[1:2]
            train_imgs = plist[2:] if len(plist) > 2 else plist

        # Write valid & test sets
        for split_name, img_sublist in [("valid", val_imgs), ("test", test_imgs)]:
            for p in img_sublist:
                dst_img = DST_DIR / split_name / "images" / f"{p.stem}_{pid}{p.suffix}"
                dst_lbl = DST_DIR / split_name / "labels" / f"{dst_img.stem}.txt"
                shutil.copy2(p, dst_img)
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    xc, yc, bw, bh = DEFAULT_PROV_BOX
                    f.write(f"{pid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                total_counts[split_name] += 1

        # Train set processing: cap if too large, augment if too small
        if len(train_imgs) > MAX_SAMPLES:
            chosen_train = random.sample(train_imgs, MAX_SAMPLES)
        else:
            chosen_train = list(train_imgs)

        for p in chosen_train:
            dst_img = DST_DIR / "train" / "images" / f"{p.stem}_{pid}{p.suffix}"
            dst_lbl = DST_DIR / "train" / "labels" / f"{dst_img.stem}.txt"
            shutil.copy2(p, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as f:
                xc, yc, bw, bh = DEFAULT_PROV_BOX
                f.write(f"{pid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            total_counts["train"] += 1

        # Augment if < TARGET_MIN_SAMPLES
        needed = TARGET_MIN_SAMPLES - len(chosen_train)
        if needed > 0:
            aug_idx = 0
            while needed > 0:
                src_p = chosen_train[aug_idx % len(chosen_train)]
                img_mat = cv2.imread(str(src_p))
                if img_mat is None:
                    aug_idx += 1
                    continue

                aug_mat, aug_box = augment_plate_and_box(img_mat, [pid] + DEFAULT_PROV_BOX, aug_idx)
                aug_name = f"{src_p.stem}_aug{aug_idx}_{pid}.jpg"
                dst_img = DST_DIR / "train" / "images" / aug_name
                dst_lbl = DST_DIR / "train" / "labels" / f"{dst_img.stem}.txt"

                cv2.imwrite(str(dst_img), aug_mat, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    f.write(f"{aug_box[0]} {aug_box[1]:.6f} {aug_box[2]:.6f} {aug_box[3]:.6f} {aug_box[4]:.6f}\n")

                total_counts["train"] += 1
                needed -= 1
                aug_idx += 1

    # Write data.yaml
    lao_yaml = {
        "path": str(DST_DIR.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": num_classes,
        "names": class_names,
    }
    with open(DST_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(lao_yaml, f, sort_keys=False, allow_unicode=True)

    print("\n" + "=" * 70)
    print("✅ Balanced Lao Province Dataset Created Successfully!")
    print(f"📊 Train Images: {total_counts['train']}")
    print(f"📊 Valid Images: {total_counts['valid']}")
    print(f"📊 Test Images:  {total_counts['test']}")
    print(f"📁 Dataset YAML: {DST_DIR / 'data.yaml'}")
    print("=" * 70)


if __name__ == "__main__":
    prepare_dataset()
