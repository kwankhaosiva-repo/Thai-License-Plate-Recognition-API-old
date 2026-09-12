"""
src/train_plate_quad.py

Trains PlateOBBNet (1-Model End-to-End License Plate Rotated Bounding Box Detector)
on datasets/Thai/LPR 2 - Polygon.yolov11_new.

Features:
- Resolution: 640x640 input resolution for high-detail detection of distant car plates.
- Head: Rigid Rotated Bounding Box (OBB) using orthogonal directional unit vectors (u, v) and (w, h).
- Loss: CenterNet Heatmap Focal Loss + WH Smooth L1 + Angle Direction Loss + Geometric Corner Loss.
- Output: Strict rectangular geometry with 100% parallel edges (no trapezoid skew).
- License: 100% BSD-3 / Apache-2.0 Compatible.
"""

import os
import sys
import math
import random
import argparse
from pathlib import Path
from typing import Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models_plate_quad import PlateQuadNet, PlateQuadNetDeploy, PlateOBBLoss

DATA_ROOT = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_SAVE_PATH = WEIGHTS_DIR / "plate_quad_detector.pth"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int, k: float = 1.0):
    """Draws a 2D Gaussian heatmap centered at `center` with `radius`."""
    diameter = 2 * radius + 1
    gaussian = np.zeros((diameter, diameter), dtype=np.float32)
    sigma = diameter / 6.0
    for y in range(diameter):
        for x in range(diameter):
            d2 = (x - radius) ** 2 + (y - radius) ** 2
            gaussian[y, x] = math.exp(-d2 / (2 * sigma * sigma))

    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


class PlateQuadDataset(Dataset):
    def __init__(self, split_dir: Path, target_size: int = 640, is_train: bool = True):
        self.split_dir = split_dir
        self.target_size = target_size
        self.is_train = is_train
        self.stride = 4
        self.feat_size = target_size // self.stride

        img_dir = self.split_dir / "images"
        lbl_dir = self.split_dir / "labels"

        self.samples = []
        if not img_dir.exists():
            return

        for img_p in sorted(img_dir.glob("*.*")):
            if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            # Read polygon labels (0 x1 y1 x2 y2 x3 y3 x4 y4)
            quads = []
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[0] == "0":
                        coords = [float(v) for v in parts[1:9]]
                        quads.append(coords)

            if quads:
                self.samples.append((img_p, quads))

        print(f"[{'Train' if is_train else 'Valid'}] Loaded {len(self.samples)} images from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, quads = self.samples[idx]
        img = cv2.imread(str(img_p))
        if img is None:
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            quads = [[0.2, 0.4, 0.6, 0.4, 0.6, 0.6, 0.2, 0.6]]

        # Resize image to target_size (640x640)
        resized = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # Photometric augmentations if training
        if self.is_train:
            alpha = random.uniform(0.75, 1.25)
            beta = random.uniform(-25, 25)
            resized = np.clip(alpha * resized + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.20:
                k = random.choice([3, 5])
                resized = cv2.GaussianBlur(resized, (k, k), 0)

        # Convert to tensor (Normalize ImageNet)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor_img = torch.from_numpy((img_rgb - mean) / std).permute(2, 0, 1).float()

        # Build Targets
        target_hm = np.zeros((1, self.feat_size, self.feat_size), dtype=np.float32)
        target_offset = np.zeros((2, self.feat_size, self.feat_size), dtype=np.float32)
        target_wh = np.zeros((2, self.feat_size, self.feat_size), dtype=np.float32)
        target_vec = np.zeros((2, self.feat_size, self.feat_size), dtype=np.float32)
        target_corners = np.zeros((8, self.feat_size, self.feat_size), dtype=np.float32)
        mask = np.zeros((self.feat_size, self.feat_size), dtype=np.float32)

        gt_quad_eval = []

        for quad in quads:
            x1, y1, x2, y2, x3, y3, x4, y4 = quad
            # Center of the 4 points
            cx = (x1 + x2 + x3 + x4) / 4.0
            cy = (y1 + y2 + y3 + y4) / 4.0

            # Width vector: average of top (P0->P1) and bottom (P3->P2)
            w_top = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            w_bot = np.array([x3 - x4, y3 - y4], dtype=np.float32)
            w_vec = 0.5 * (w_top + w_bot)
            w_len = float(np.linalg.norm(w_vec))
            w_len = max(w_len, 0.01)

            # Height vector: average of left (P0->P3) and right (P1->P2)
            h_left = np.array([x4 - x1, y4 - y1], dtype=np.float32)
            h_right = np.array([x3 - x2, y3 - y2], dtype=np.float32)
            h_vec = 0.5 * (h_left + h_right)
            h_len = float(np.linalg.norm(h_vec))
            h_len = max(h_len, 0.01)

            # Unit directional vector u = [ux, uy] pointing along width
            u_vec = w_vec / w_len

            # Grid coordinates (0 to feat_size - 1)
            gx = cx * self.feat_size
            gy = cy * self.feat_size
            ix, iy = int(gx), int(gy)

            if 0 <= ix < self.feat_size and 0 <= iy < self.feat_size:
                pw = w_len * self.feat_size
                ph = h_len * self.feat_size
                radius = max(2, int(min(pw, ph) / 2))

                draw_gaussian(target_hm[0], (ix, iy), radius)

                # Subpixel center offset in grid space
                target_offset[0, iy, ix] = gx - ix
                target_offset[1, iy, ix] = gy - iy

                # Dimensions (w, h) in normalized image space
                target_wh[0, iy, ix] = w_len
                target_wh[1, iy, ix] = h_len

                # Orientation unit vector [ux, uy]
                target_vec[0, iy, ix] = u_vec[0]
                target_vec[1, iy, ix] = u_vec[1]

                # GT Corners for geometric corner loss
                target_corners[0, iy, ix] = x1
                target_corners[1, iy, ix] = y1
                target_corners[2, iy, ix] = x2
                target_corners[3, iy, ix] = y2
                target_corners[4, iy, ix] = x3
                target_corners[5, iy, ix] = y3
                target_corners[6, iy, ix] = x4
                target_corners[7, iy, ix] = y4

                mask[iy, ix] = 1.0
                gt_quad_eval.append([x1, y1, x2, y2, x3, y3, x4, y4])

        return (
            tensor_img,
            torch.from_numpy(target_hm),
            torch.from_numpy(target_offset),
            torch.from_numpy(target_wh),
            torch.from_numpy(target_vec),
            torch.from_numpy(target_corners),
            torch.from_numpy(mask),
            torch.tensor(gt_quad_eval[0] if gt_quad_eval else [0] * 8, dtype=torch.float32),
        )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, target_size: int = 640):
    model.eval()
    criterion = PlateOBBLoss()
    deploy_wrapper = PlateQuadNetDeploy(model, topk=1).to(device)

    total_loss = 0.0
    total_samples = 0
    corner_pixel_errors = []

    with torch.no_grad():
        for imgs, t_hm, t_off, t_wh, t_vec, t_corn, mask, gt_quads in loader:
            imgs = imgs.to(device)
            t_hm = t_hm.to(device)
            t_off = t_off.to(device)
            t_wh = t_wh.to(device)
            t_vec = t_vec.to(device)
            t_corn = t_corn.to(device)
            mask = mask.to(device)

            preds = model(imgs)
            loss, _, _, _, _ = criterion(preds, t_hm, t_off, t_wh, t_vec, t_corn, mask)
            total_loss += loss.item() * len(imgs)
            total_samples += len(imgs)

            # Evaluate corner error using Deploy wrapper
            pred_quads, scores = deploy_wrapper(imgs)  # pred_quads: (B, 1, 4, 2)
            pred_pts = pred_quads[:, 0, :, :].cpu().numpy()  # (B, 4, 2)
            gt_pts = gt_quads.numpy().reshape(-1, 4, 2)       # (B, 4, 2)

            for b in range(len(imgs)):
                if scores[b, 0] > 0.1:
                    err = np.linalg.norm((pred_pts[b] - gt_pts[b]) * target_size, axis=1).mean()
                    corner_pixel_errors.append(err)

    val_loss = total_loss / max(total_samples, 1)
    mean_corner_err = float(np.mean(corner_pixel_errors)) if corner_pixel_errors else 999.0
    return val_loss, mean_corner_err


def train_plate_quad(epochs: int = 35, batch_size: int = 8, lr: float = 1e-3, target_size: int = 640):
    print("=" * 70)
    print("🚀 Starting PlateOBBNet Training (1-Model Rigid OBB Detector at 640x640)")
    print(f"Device      : {DEVICE}")
    print(f"Dataset Root: {DATA_ROOT}")
    print(f"Resolution  : {target_size}x{target_size}")
    print(f"Target Save : {MODEL_SAVE_PATH}")
    print("=" * 70)

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_ROOT}")

    # Datasets & Loaders
    train_ds = PlateQuadDataset(DATA_ROOT / "train", target_size=target_size, is_train=True)
    val_ds = PlateQuadDataset(DATA_ROOT / "valid", target_size=target_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model & Loss
    model = PlateQuadNet(pretrained=True).to(DEVICE)
    criterion = PlateOBBLoss(wh_weight=2.0, vec_weight=2.0, corner_weight=6.0, offset_weight=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_corner_err = float("inf")
    best_epoch = 0

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_hm_loss = 0.0
        train_cr_loss = 0.0
        train_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for imgs, t_hm, t_off, t_wh, t_vec, t_corn, mask, _ in pbar:
            imgs = imgs.to(DEVICE)
            t_hm = t_hm.to(DEVICE)
            t_off = t_off.to(DEVICE)
            t_wh = t_wh.to(DEVICE)
            t_vec = t_vec.to(DEVICE)
            t_corn = t_corn.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, l_hm, l_cr, l_wh, l_vec = criterion(preds, t_hm, t_off, t_wh, t_vec, t_corn, mask)
            loss.backward()
            optimizer.step()

            bs = len(imgs)
            train_loss += loss.item() * bs
            train_hm_loss += l_hm.item() * bs
            train_cr_loss += l_cr.item() * bs
            train_samples += bs

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "hm": f"{l_hm.item():.4f}",
                "cr": f"{l_cr.item():.4f}",
                "wh": f"{l_wh.item():.4f}",
            })

        scheduler.step()

        # Validation
        val_loss, val_corner_err = evaluate(model, val_loader, DEVICE, target_size=target_size)
        print(f"--> [Epoch {epoch:02d}] Train Loss: {train_loss/train_samples:.4f} | Val Loss: {val_loss:.4f} | Mean Corner Error: {val_corner_err:.2f} px")

        # Save Best Checkpoint
        if val_corner_err < best_corner_err:
            best_corner_err = val_corner_err
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_corner_err": best_corner_err,
                "target_size": target_size,
                "license": "BSD-3 / Apache-2.0 Compatible",
            }, MODEL_SAVE_PATH)
            print(f"    ⭐ New Best Checkpoint saved to {MODEL_SAVE_PATH} (Corner Error: {best_corner_err:.2f} px)")

    print("\n" + "=" * 70)
    print(f"🎉 Training Complete! Best Checkpoint at Epoch {best_epoch} with Mean Corner Error: {best_corner_err:.2f} px")
    print(f"Saved weights: {MODEL_SAVE_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    train_plate_quad(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, target_size=args.size)
