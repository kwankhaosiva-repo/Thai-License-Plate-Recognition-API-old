"""
src/train_plate_corner_regressor.py

Trains a lightweight, 100% Permissive (BSD-3-Clause / Apache-2.0) MobileNetV3-Small
network to regress the 4 corner points [x1, y1, x2, y2, x3, y3, x4, y4] of tilted license
plates from cropped bounding box detections.

Dataset:
  datasets/Thai/LPR 2 - Polygon.yolov11_new (1,068 train images, 305 val images)
  Format: 0 x1 y1 x2 y2 x3 y3 x4 y4 (normalized 4-corner polygon)

Output:
  weights/plate_corner_regressor.pth
"""

import os
import sys
import math
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg

WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = WEIGHTS_DIR / "plate_corner_regressor.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 quadrilateral 2D points clockwise starting from Top-Left:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


class PlateCornerDataset(Dataset):
    def __init__(self, data_split_dir: Path, is_train: bool = True, target_size: int = 224):
        self.is_train = is_train
        self.target_size = target_size
        self.samples = []

        images_dir = data_split_dir / "images"
        labels_dir = data_split_dir / "labels"

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        label_files = list(labels_dir.glob("*.txt"))

        for lbl_p in label_files:
            stem = lbl_p.stem
            # Find matching image
            img_p = None
            for ext in valid_exts:
                cand = images_dir / f"{stem}{ext}"
                if cand.exists():
                    img_p = cand
                    break
            if img_p is None:
                continue

            # Read polygon points
            with open(lbl_p, "r") as f:
                line = f.readline().strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:  # class + 8 coords
                continue

            try:
                coords = [float(x) for x in parts[1:9]]
                pts = np.array(coords, dtype=np.float32).reshape(4, 2)
                self.samples.append((str(img_p), pts))
            except Exception:
                continue

        print(f"[{'Train' if is_train else 'Valid'}] Loaded {len(self.samples)} valid 4-corner polygon samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, norm_pts = self.samples[idx]
        img = cv2.imread(img_p)
        if img is None:
            # Fallback black image
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            norm_pts = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]], dtype=np.float32)

        h, w = img.shape[:2]
        pixel_pts = norm_pts.copy()
        pixel_pts[:, 0] *= w
        pixel_pts[:, 1] *= h
        pixel_pts = order_quad_clockwise(pixel_pts)

        # Compute bounding box of the polygon
        min_x = pixel_pts[:, 0].min()
        max_x = pixel_pts[:, 0].max()
        min_y = pixel_pts[:, 1].min()
        max_y = pixel_pts[:, 1].max()

        bw = max(10.0, max_x - min_x)
        bh = max(10.0, max_y - min_y)

        # Margin expansion to simulate real RT-DETR detection box variations
        if self.is_train:
            pad_left = random.uniform(0.04, 0.22) * bw
            pad_right = random.uniform(0.04, 0.22) * bw
            pad_top = random.uniform(0.04, 0.22) * bh
            pad_bottom = random.uniform(0.04, 0.22) * bh
        else:
            pad_left = 0.10 * bw
            pad_right = 0.10 * bw
            pad_top = 0.10 * bh
            pad_bottom = 0.10 * bh

        crop_x1 = max(0, int(min_x - pad_left))
        crop_y1 = max(0, int(min_y - pad_top))
        crop_x2 = min(w, int(max_x + pad_right))
        crop_y2 = min(h, int(max_y + pad_bottom))

        cw = max(1, crop_x2 - crop_x1)
        ch = max(1, crop_y2 - crop_y1)

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            crop = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            cw, ch = self.target_size, self.target_size
            crop_x1, crop_y1 = 0, 0

        # Transform 4 corner points into local crop coordinate frame [0, 1]
        local_pts = pixel_pts.copy()
        local_pts[:, 0] = np.clip((local_pts[:, 0] - crop_x1) / float(cw), 0.0, 1.0)
        local_pts[:, 1] = np.clip((local_pts[:, 1] - crop_y1) / float(ch), 0.0, 1.0)

        # Resize crop to target size
        crop_resized = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # Color augmentations in training
        if self.is_train and random.random() < 0.5:
            # Random brightness & contrast
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-15, 15)
            crop_resized = np.clip(alpha * crop_resized + beta, 0, 255).astype(np.uint8)

        # Convert to float tensor normalized [0, 1], with ImageNet mean/std
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        tensor_img = transforms.ToTensor()(crop_rgb)
        tensor_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_img)

        # Target: 8 normalized values [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
        target_coords = torch.tensor(local_pts.flatten(), dtype=torch.float32)

        return tensor_img, target_coords


def build_corner_model() -> nn.Module:
    base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = base.classifier[0].in_features
    base.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.Hardswish(),
        nn.Dropout(0.1),
        nn.Linear(128, 8),
        nn.Sigmoid(),  # Coordinates strictly in range [0, 1]
    )
    return base


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_pixel_err = 0.0
    count = 0
    criterion = nn.SmoothL1Loss(beta=0.02)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = criterion(preds, targets)
            total_loss += loss.item() * len(targets)

            # Measure Mean Pixel Error (assuming 224x224 crop)
            diff = (preds - targets).view(-1, 4, 2) * 224.0
            dist = torch.norm(diff, dim=-1).mean(dim=-1)
            total_pixel_err += dist.sum().item()
            count += len(targets)

    if count == 0:
        return 0.0, 0.0
    return total_loss / count, total_pixel_err / count


def train_corner_regressor(epochs: int = 30, batch_size: int = 32, lr: float = 1e-3):
    print("==================================================================")
    print("--- Training Plate 4-Corner Homography Regressor (MobileNetV3) ---")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")
    print("==================================================================")

    data_dir = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new"
    train_ds = PlateCornerDataset(data_dir / "train", is_train=True, target_size=224)
    val_ds = PlateCornerDataset(data_dir / "valid", is_train=False, target_size=224)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_corner_model().to(DEVICE)
    criterion = nn.SmoothL1Loss(beta=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_pixel_err = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for imgs, targets in pbar:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            train_count += len(targets)
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        scheduler.step()
        train_avg_loss = train_loss / train_count

        val_loss, val_pix_err = evaluate(model, val_loader, DEVICE)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_avg_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | Val Corner Error: {val_pix_err:.2f} px (224x224)"
        )

        if val_pix_err < best_pixel_err:
            best_pixel_err = val_pix_err
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_pix_err": val_pix_err,
                "val_loss": val_loss,
                "architecture": "MobileNetV3-Small-Corner-Regressor",
                "license": "BSD-3-Clause",
            }, str(SAVE_PATH))
            print(f"   --> New Best Checkpoint saved! (Corner Error: {val_pix_err:.2f} px)")

    print("\n==================================================================")
    print("Training Complete!")
    print(f"Best Epoch: {best_epoch} with Mean Corner Error: {best_pixel_err:.2f} pixels")
    print(f"Saved Checkpoint: {SAVE_PATH}")
    print("==================================================================")


if __name__ == "__main__":
    train_corner_regressor(epochs=25, batch_size=32, lr=1e-3)
