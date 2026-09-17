"""
src/train_grayscale_province_thai.py

Trains Model 3B Thai Province Classifier using the Grayscale Image Classification approach:
1. Converts all crops to Grayscale to be completely invariant to plate background color (yellow truck, white car, green rental).
2. Uses GrayscaleSmartResize((256, 64)) which dynamically fills padding using the plate's genuine background tone (instead of artificial pitch-black bars).
3. Preserves weights/province_model.pth intact.
4. Saves best model to weights/province_model_grayscale_thai.pth.
"""

import os
import shutil
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models import ResNetProvinceClassifier

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DATA_DIR = PROJECT_ROOT / "datasets" / "Thai" / "thai_province_crops"
PROV_MAP_PATH = PROJECT_ROOT / "weights" / "province_map.json"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_SAVE_PATH = WEIGHTS_DIR / "province_model_grayscale_thai.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class GrayscaleSmartResize:
    def __init__(self, size=(256, 64)):
        self.target_w, self.target_h = size

    def __call__(self, img_pil: Image.Image):
        gray_im = img_pil.convert("L")
        w, h = gray_im.size

        im_arr = np.array(gray_im)
        corners = [im_arr[0, 0], im_arr[0, -1], im_arr[-1, 0], im_arr[-1, -1]]
        bg_val = int(np.median(corners))

        ratio = min(self.target_w / w, self.target_h / h)
        new_w, new_h = max(4, int(w * ratio)), max(4, int(h * ratio))
        resized = gray_im.resize((new_w, new_h), Image.BICUBIC)

        new_im = Image.new("L", (self.target_w, self.target_h), color=bg_val)
        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        new_im.paste(resized, (paste_x, paste_y))

        return new_im.convert("RGB")


def get_grayscale_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            GrayscaleSmartResize((256, 64)),
            transforms.RandomAffine(degrees=4, translate=(0.02, 0.04)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAutocontrast(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])
    else:
        return transforms.Compose([
            GrayscaleSmartResize((256, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])


class ThaiProvinceDataset(Dataset):
    def __init__(self, split_dir, class_to_idx, transform=None):
        self.samples = []
        self.transform = transform
        self.split_dir = Path(split_dir)

        if not self.split_dir.exists():
            return

        for folder in sorted(self.split_dir.iterdir()):
            if folder.is_dir() and folder.name in class_to_idx:
                label = class_to_idx[folder.name]
                for img_p in folder.glob("*.jpg"):
                    self.samples.append((img_p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, label = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(samples, n_classes=77):
    counts = np.zeros(n_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    total = float(len(samples))
    weights = (total / (n_classes * np.maximum(counts, 1.0))) ** 0.5
    weights = np.clip(weights, 0.2, 8.0)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, top_k=5):
    model.eval()
    val_loss = 0.0
    val_correct_1 = 0
    val_correct_k = 0
    val_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * len(labels)
            preds1 = outputs.argmax(dim=1)
            val_correct_1 += (preds1 == labels).sum().item()

            _, topk = outputs.topk(min(top_k, outputs.size(1)), dim=1)
            val_correct_k += sum([labels[i].item() in topk[i] for i in range(len(labels))])
            val_total += len(labels)

    if val_total == 0:
        return 0.0, 0.0, 0.0

    return val_loss / val_total, val_correct_1 / val_total, val_correct_k / val_total


def train_grayscale_thai(epochs=12, batch_size=32, lr=2e-4):
    print("=" * 70)
    print("🇹🇭 Training Thai Province Grayscale Classifier (77 Classes)")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATA_DIR}")
    print(f"Target Checkpoint: {MODEL_SAVE_PATH}")
    print("=" * 70)

    with open(PROV_MAP_PATH, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    n_classes = len(prov_map)

    class_to_idx = {f"{int(k):02d}_{v}": int(k) for k, v in prov_map.items()}

    train_tf = get_grayscale_transforms(is_train=True)
    val_tf = get_grayscale_transforms(is_train=False)

    train_ds = ThaiProvinceDataset(DATA_DIR / "train", class_to_idx, transform=train_tf)
    val_ds = ThaiProvinceDataset(DATA_DIR / "valid", class_to_idx, transform=val_tf)
    test_ds = ThaiProvinceDataset(DATA_DIR / "test", class_to_idx, transform=val_tf)

    print(f"Dataset splits: Train={len(train_ds)}, Valid={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = ResNetProvinceClassifier(n_classes=n_classes, backbone="resnet18", pretrained=True)
    model = model.to(DEVICE)

    class_weights = compute_class_weights(train_ds.samples, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_top1 = 0.0
    best_val_top5 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{train_correct / train_total * 100:.1f}%"})

        scheduler.step()

        val_loss, val_top1, val_top5 = evaluate(model, val_loader, DEVICE, top_k=5)
        print(f"  Validation -> Loss: {val_loss:.4f} | Top-1: {val_top1 * 100:.2f}% | Top-5: {val_top5 * 100:.2f}%")

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top5 = val_top5
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "n_classes": n_classes,
                "epoch": epoch,
                "val_top1": val_top1,
                "val_top5": val_top5,
            }, MODEL_SAVE_PATH)
            print(f"  🏆 New best Top-1: {val_top1 * 100:.2f}% -> Saved to {MODEL_SAVE_PATH.name}")

    # Evaluate best model on test set
    if MODEL_SAVE_PATH.exists():
        ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        test_loss, test_top1, test_top5 = evaluate(model, test_loader, DEVICE, top_k=5)
        print("\n" + "=" * 70)
        print(f"🏁 Final Test Set Evaluation:")
        print(f"  Top-1 Accuracy: {test_top1 * 100:.2f}%")
        print(f"  Top-5 Accuracy: {test_top5 * 100:.2f}%")
        print(f"  Weights saved: {MODEL_SAVE_PATH}")
        print("=" * 70)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Thai Province Grayscale ResNet18 Classifier (77 classes)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    args = parser.parse_args()

    train_grayscale_thai(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
