"""
src/train_lao_character_classifier.py

Trains the Lao Character Classifier (MobileNetV2, BSD-3) on balanced character crops.
Automatically exports the best checkpoint to standalone ONNX (Opset 18) for C# deployment.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

try:
    import onnx
except ImportError:
    onnx = None

BASE_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_character_crops"
TRAIN_DIR = BASE_DIR / "train"
VALID_DIR = BASE_DIR / "valid"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MAP_PATH = WEIGHTS_DIR / "char_classifier_map_lao.json"
MODEL_SAVE_PATH = WEIGHTS_DIR / "character_classifier_lao.pth"
ONNX_SAVE_PATH = WEIGHTS_DIR / "character_classifier_lao_opset18.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class LaoCharacterDataset(Dataset):
    def __init__(self, root_dir: Path, class_to_idx: dict, transform=None):
        self.samples = []
        self.transform = transform
        self.root_dir = Path(root_dir)

        for folder in sorted(self.root_dir.iterdir()):
            if folder.is_dir() and folder.name in class_to_idx:
                label = class_to_idx[folder.name]
                img_list = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                for img_p in img_list:
                    self.samples.append((img_p, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_p, label = self.samples[idx]
        img = Image.open(img_p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def compute_class_weights(samples, n_classes):
    counts = np.zeros(n_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    total = float(len(samples))
    weights = (total / (n_classes * np.maximum(counts, 1.0))) ** 0.5
    weights = np.clip(weights, 0.4, 3.0)
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


def export_to_onnx(model: nn.Module, onnx_path: Path, n_classes: int, opset: int = 18):
    """Exports the trained classifier to a standalone ONNX model with embedded weights."""
    print(f"\n📦 Exporting Lao Character Classifier to ONNX (Opset {opset})...")
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64, device="cpu", dtype=torch.float32)
    cpu_model = model.cpu()

    # Wrap with Softmax for probability outputs
    class SoftmaxWrapper(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
        def forward(self, x):
            return F.softmax(self.core(x), dim=1)

    export_model = SoftmaxWrapper(cpu_model)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
    )

    if onnx is not None:
        from onnx.external_data_helper import load_external_data_for_model
        model_proto = onnx.load(str(onnx_path))
        load_external_data_for_model(model_proto, str(onnx_path.parent))
        onnx.save(model_proto, str(onnx_path), save_as_external_data=False)
        onnx.checker.check_model(model_proto)

        data_file = onnx_path.with_name(f"{onnx_path.name}.data")
        if data_file.exists():
            data_file.unlink()

        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  --> ONNX Model validated successfully! Standalone size: {size_mb:.2f} MB")
        print(f"  --> Saved to: {onnx_path}")


def train_lao_character_classifier(epochs=15, batch_size=64, lr=4e-4):
    print(f"\n=======================================================")
    print(f"--- Training Lao Character Classifier (MobileNetV2) ---")
    print(f"Device     : {DEVICE}")
    print(f"Epochs     : {epochs}, Batch Size: {batch_size}, LR: {lr}")
    print(f"Target Save: {MODEL_SAVE_PATH}")
    print(f"=======================================================\n")

    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Class map not found: {MAP_PATH}")

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_char = json.load(f)
    n_classes = len(idx_to_char)
    class_to_idx = {v: int(k) for k, v in idx_to_char.items()}
    print(f"Total Lao Character Classes: {n_classes}")

    # 1. Transforms (Input 64x64)
    train_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomAffine(degrees=6, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.ColorJitter(brightness=0.20, contrast=0.20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Datasets & Loaders
    train_ds = LaoCharacterDataset(TRAIN_DIR, class_to_idx, transform=train_tf)
    val_ds = LaoCharacterDataset(VALID_DIR, class_to_idx, transform=val_tf)

    print(f"Loaded: Train = {len(train_ds)} samples, Valid = {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model Architecture: MobileNetV2 with n_classes outputs
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, n_classes)
    )
    model = model.to(DEVICE)

    # 4. Class weights & Optimizer
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
                "n_classes": n_classes,
                "license": "BSD-3 / Apache-2.0 Compatible",
            }, str(MODEL_SAVE_PATH))
            print(f"   --> ⭐ New Best Checkpoint saved! (Val Top-1: {val_top1*100:.2f}%, Top-3: {val_top3*100:.2f}%)")

    print(f"\n=======================================================")
    print(f"🎉 Training Complete! Best Checkpoint at Epoch {best_epoch}")
    print(f"Best Validation Top-1 Accuracy: {best_val_top1*100:.2f}%")
    print(f"Best Validation Top-3 Accuracy: {best_val_top3*100:.2f}%")
    print(f"Saved checkpoint to: {MODEL_SAVE_PATH}")
    print(f"=======================================================")

    # Export best model to Standalone ONNX
    best_ckpt = torch.load(MODEL_SAVE_PATH, map_location="cpu")
    best_model = models.mobilenet_v2(weights=None)
    best_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(best_model.last_channel, n_classes)
    )
    best_model.load_state_dict(best_ckpt["model_state"])
    export_to_onnx(best_model, ONNX_SAVE_PATH, n_classes=n_classes, opset=18)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()

    train_lao_character_classifier(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
