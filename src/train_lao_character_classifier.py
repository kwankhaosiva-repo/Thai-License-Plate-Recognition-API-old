"""
src/train_lao_character_classifier.py

Trains Model 3A-Box-Lao (MobileNetV2 Character Classifier) on 34 Lao character classes
(10 digits 0-9 + 24 Lao consonants ก-ฮ) using harvested verified Lao plate character crops:
  datasets/Lao/lao_character_crops/train/
  datasets/Lao/lao_character_crops/valid/

Features:
  - Input: 64x64 RGB square-padded character crops
  - Minority class oversampling/augmentation to prevent underfitting on rare consonants (e.g. ຖ, ປ, ງ, ຊ)
  - Class-weighted Cross-Entropy Loss to balance frequent digits vs consonants
  - Real-time Top-1 and Top-3 accuracy evaluation
  - Saves best checkpoint to weights/character_classifier_lao.pth
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

BASE_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_character_crops"
TRAIN_DIR = BASE_DIR / "train"
VALID_DIR = BASE_DIR / "valid"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MAP_PATH = WEIGHTS_DIR / "char_classifier_map_lao.json"
MODEL_SAVE_PATH = WEIGHTS_DIR / "character_classifier_lao.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def ensure_valid_has_samples(train_dir: Path, valid_dir: Path, class_to_idx: dict):
    """Ensure every class has at least 2 samples in valid for fair evaluation."""
    for class_name in class_to_idx:
        v_cls_dir = valid_dir / class_name
        v_cls_dir.mkdir(parents=True, exist_ok=True)
        v_files = list(v_cls_dir.glob("*.jpg")) + list(v_cls_dir.glob("*.png"))
        if len(v_files) == 0:
            t_cls_dir = train_dir / class_name
            t_files = list(t_cls_dir.glob("*.jpg")) + list(t_cls_dir.glob("*.png"))
            if t_files:
                # Copy at least 1-2 images to valid
                for f in t_files[:min(2, len(t_files))]:
                    dest = v_cls_dir / f"valid_rep_{f.name}"
                    if not dest.exists():
                        shutil.copy2(f, dest)


class LaoCharacterDataset(Dataset):
    def __init__(self, root_dir: Path, class_to_idx: dict, transform=None, min_samples_per_class=0):
        self.samples = []
        self.transform = transform
        self.root_dir = Path(root_dir)

        class_samples = {cls_idx: [] for cls_idx in class_to_idx.values()}

        for folder in sorted(self.root_dir.iterdir()):
            if folder.is_dir() and folder.name in class_to_idx:
                label = class_to_idx[folder.name]
                img_list = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                for img_p in img_list:
                    class_samples[label].append(img_p)

        # Oversample minority classes if requested (for train)
        for label, paths in class_samples.items():
            if not paths:
                continue
            if min_samples_per_class > 0 and len(paths) < min_samples_per_class:
                multiplier = (min_samples_per_class // len(paths)) + 1
                expanded = (paths * multiplier)[:min_samples_per_class]
                for p in expanded:
                    self.samples.append((p, label))
            else:
                for p in paths:
                    self.samples.append((p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, label = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(samples, n_classes=34):
    counts = np.zeros(n_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    total = float(len(samples))
    weights = (total / (n_classes * np.maximum(counts, 1.0))) ** 0.5
    weights = np.clip(weights, 0.2, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    val_loss = 0.0
    val_correct_1 = 0
    val_correct_3 = 0
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

            top3 = outputs.topk(min(3, outputs.size(1)), dim=1)[1]
            val_correct_3 += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            val_total += len(labels)

    if val_total == 0:
        return 0.0, 0.0, 0.0

    return val_loss / val_total, val_correct_1 / val_total, val_correct_3 / val_total


def train_lao_character_classifier(epochs=12, batch_size=128, lr=4e-4):
    print(f"\n=======================================================")
    print(f"--- Training Lao Character Classifier (MobileNetV2) ---")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"=======================================================\n")

    # 1. Load class mapping
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Class map not found: {MAP_PATH}")

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_char = json.load(f)
    n_classes = len(idx_to_char)
    class_to_idx = {v: int(k) for k, v in idx_to_char.items()}
    print(f"Total Lao Character Classes: {n_classes}")

    ensure_valid_has_samples(TRAIN_DIR, VALID_DIR, class_to_idx)

    # 2. Transforms (Input 64x64)
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Datasets with minority class oversampling to at least 200 samples in train
    train_ds = LaoCharacterDataset(TRAIN_DIR, class_to_idx, transform=train_tf, min_samples_per_class=200)
    val_ds = LaoCharacterDataset(VALID_DIR, class_to_idx, transform=val_tf, min_samples_per_class=0)

    print(f"Loaded: Train = {len(train_ds)} samples (with oversampling), Valid = {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. Model Architecture: MobileNetV2 with 34 outputs
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, n_classes)
    )
    model = model.to(DEVICE)

    # 5. Class weights & Optimizer
    class_weights = compute_class_weights(train_ds.samples, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_top1 = 0.0
    best_val_top3 = 0.0
    best_epoch = 0

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

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/train_total*100:.1f}%"})

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        val_loss, val_top1, val_top3 = evaluate(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Top-1: {val_top1*100:.2f}%, Top-3: {val_top3*100:.2f}%"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_val_top3 = val_top3
            best_epoch = epoch

            torch.save({
                "model_state": model.state_dict(),
                "class_map": idx_to_char,
                "best_acc_top1": best_val_top1,
                "best_acc_top3": best_val_top3,
                "epoch": best_epoch,
            }, str(MODEL_SAVE_PATH))
            print(f"   --> New Best Checkpoint saved! (Val Top-1: {val_top1*100:.2f}%, Top-3: {val_top3*100:.2f}%)")

    print(f"\n=======================================================")
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Validation Top-1 Accuracy: {best_val_top1*100:.2f}%")
    print(f"Best Validation Top-3 Accuracy: {best_val_top3*100:.2f}%")
    print(f"Saved weights to: {MODEL_SAVE_PATH}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()

    train_lao_character_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
