"""
src/balance_and_augment_split_dataset.py

Performs targeted offline data augmentation directly on:
  datasets/Thai/thai_character_crops/splits/train/
  datasets/Thai/thai_character_crops/splits/valid/

Preserves all existing user-curated ground truth files intact.
Brings all rare Thai consonants up to balanced production representation
(at least 400 samples in train, 30 samples in valid) with rich photometric variations
(including dark bumper shadows, low contrast, blur, and affine perspective shifts).
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "datasets" / "Thai" / "thai_character_crops" / "splits"
TRAIN_DIR = SPLITS_DIR / "train"
VALID_DIR = SPLITS_DIR / "valid"

TARGET_TRAIN_SAMPLES = 400
TARGET_VALID_SAMPLES = 30


def apply_photometric_shadow(im: Image.Image) -> Image.Image:
    """Applies realistic lighting, shadow gradients, or contrast drops."""
    arr = np.array(im, dtype=np.float32)
    h, w = arr.shape[:2]

    style = random.choice(["uniform_dark", "gradient_top", "gradient_bottom", "gradient_left", "gradient_right", "high_contrast", "none"])

    if style == "uniform_dark":
        factor = random.uniform(0.35, 0.65)
        arr = arr * factor
    elif style == "gradient_top":
        grad = np.linspace(random.uniform(0.3, 0.5), random.uniform(0.8, 1.0), h)[:, None, None]
        arr = arr * grad
    elif style == "gradient_bottom":
        grad = np.linspace(random.uniform(0.8, 1.0), random.uniform(0.3, 0.5), h)[:, None, None]
        arr = arr * grad
    elif style == "gradient_left":
        grad = np.linspace(random.uniform(0.3, 0.5), random.uniform(0.8, 1.0), w)[None, :, None]
        arr = arr * grad
    elif style == "gradient_right":
        grad = np.linspace(random.uniform(0.8, 1.0), random.uniform(0.3, 0.5), w)[None, :, None]
        arr = arr * grad

    res = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Random contrast jitter
    c_factor = random.uniform(0.55, 1.35)
    res = ImageEnhance.Contrast(res).enhance(c_factor)

    # Random slight blur or sharpness
    r = random.random()
    if r < 0.25:
        res = res.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 0.9)))
    elif r < 0.45:
        res = ImageEnhance.Sharpness(res).enhance(random.uniform(1.2, 1.8))

    return res


def apply_geometric_transform(im: Image.Image) -> Image.Image:
    """Applies slight rotation and translation."""
    angle = random.uniform(-6, 6)
    corners = [
        im.getpixel((0, 0)),
        im.getpixel((im.width - 1, 0)),
        im.getpixel((0, im.height - 1)),
        im.getpixel((im.width - 1, im.height - 1)),
    ]
    if isinstance(corners[0], tuple):
        avg_bg = tuple(int(np.mean([c[i] for c in corners])) for i in range(len(corners[0])))
    else:
        avg_bg = int(np.mean(corners))

    res = im.rotate(angle, resample=Image.BICUBIC, fillcolor=avg_bg)
    tx = random.randint(-2, 2)
    ty = random.randint(-2, 2)
    res = res.transform(res.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=avg_bg)
    return res


def balance_split(split_dir: Path, target_count: int, split_name: str):
    print(f"\n=======================================================")
    print(f"Balancing {split_name} split in: {split_dir}")
    print(f"Target minimum samples per class: {target_count}")
    print(f"=======================================================\n")

    if not split_dir.exists():
        print(f"Error: {split_dir} does not exist!")
        return

    subdirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} class folders.")

    for folder in tqdm(subdirs, desc=f"Balancing {split_name}"):
        existing_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        current_count = len(existing_files)

        if current_count == 0:
            # If valid split is missing this class, borrow from train
            train_src = TRAIN_DIR / folder.name
            if train_src.exists():
                src_files = list(train_src.glob("*.jpg")) + list(train_src.glob("*.png"))
                if src_files:
                    sample_take = min(5, len(src_files))
                    for i in range(sample_take):
                        im = Image.open(src_files[i]).convert("RGB")
                        im.save(folder / f"seed_borrow_{i}.jpg", quality=95)
                    existing_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                    current_count = len(existing_files)

        if current_count == 0:
            print(f"Warning: No images available to balance class '{folder.name}'!")
            continue

        needed = target_count - current_count
        if needed <= 0:
            continue

        gen_idx = 0
        while gen_idx < needed:
            src_p = random.choice(existing_files)
            try:
                with Image.open(src_p) as im:
                    im_rgb = im.convert("RGB")
                    aug_geom = apply_geometric_transform(im_rgb)
                    aug_final = apply_photometric_shadow(aug_geom)

                    out_name = f"aug_syn_{gen_idx:04d}_{src_p.stem[:25]}.jpg"
                    aug_final.save(folder / out_name, quality=92)
                    gen_idx += 1
            except Exception as e:
                print(f"Error augmenting {src_p}: {e}")
                break

    print(f"\nCompleted balancing {split_name} split.")


if __name__ == "__main__":
    balance_split(TRAIN_DIR, TARGET_TRAIN_SAMPLES, "train")
    balance_split(VALID_DIR, TARGET_VALID_SAMPLES, "valid")
