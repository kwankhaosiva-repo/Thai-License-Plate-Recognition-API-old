"""
src/train_lao_ocr.py

Trains a Lao OCR fallback model (ResNet18 + BiLSTM + CTC) using the EXACT same
train/serve-matched pipeline as the Thai OCR trainer (src/train_ocr.py):

  - Preprocess (train == serve, v2 audit contract):
      convert("L") -> ImageOps.autocontrast(cutoff=1) -> SmartResize((256,64), pad 0)
      -> ToTensor()  (no extra normalization)  [via src.preprocess.get_ocr_transforms]
  - Postprocess: best_path_decode (same function used at serve time)
  - Validation reports CER (Levenshtein / GT length) + Format-Valid rate
    using the Lao plate grammar (2 consonants + 1-4 digits).

Dataset: datasets/Lao/lao-plate-dataset (ground_truth_train/validation/test.csv
with columns filename,image_path,plate_text,...; splits verified leak-free by
filename, 2026-09-26). Vocab comes from weights/int_to_char_lao.json (35 tokens
incl. blank). NOTE: a small number of GT rows contain "ภ" which is NOT in the
OCR vocab — CTC simply never predicts it; rows remain usable since
OCRDataset.encode filters unknown chars, but empty encodings are dropped via
collate-safe min-length filter below.

v3tag parity with Thai trainer: --tag writes ocr_model_lao_<tag>.pth and never
resumes from the production checkpoint.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ResNetCRNN, best_path_decode
from src.preprocess import get_ocr_transforms

DATA_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao-plate-dataset"
CHAR_MAP_PATH = PROJECT_ROOT / "weights" / "int_to_char_lao.json"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_SAVE_PATH = WEIGHTS_DIR / "ocr_model_lao.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

LAO_CONSONANTS = set("ກຂຄງຈຊຍດຕຖທນບປຜພຟມຢຣລວສຫອຮ")


def levenshtein(s1: str, s2: str) -> int:
    """Standard edit distance."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def format_lao_plate(text: str) -> str:
    """Normalize Lao plate text: single space between consonant block and digits."""
    t = " ".join(str(text).split())
    letters = "".join(c for c in t if c in LAO_CONSONANTS)
    digits = "".join(c for c in t if c.isdigit())
    if letters and digits:
        return f"{letters} {digits}"
    return t


def is_valid_lao_plate(text: str) -> bool:
    """Lao standard: 2 consonants + 1-4 digits (allow 1-3 letters for edge cases)."""
    parts = text.split()
    if len(parts) != 2:
        return False
    letters, digits = parts
    return (
        1 <= len(letters) <= 3
        and all(c in LAO_CONSONANTS for c in letters)
        and 1 <= len(digits) <= 4
        and digits.isdigit()
    )


class LaoOCRDataset(torch.utils.data.Dataset):
    """Plate-crop OCR dataset for the lao-plate-dataset CSV layout
    (columns: filename, image_path, plate_text, ...). Encodes GT with the CTC
    char map, skipping chars outside the vocab; drops rows that encode empty."""

    def __init__(self, df: pd.DataFrame, root: Path, char_map: dict, transform=None):
        self.root = Path(root)
        self.char_map = char_map
        self.transform = transform
        self.samples = []
        for row in df.itertuples(index=False):
            text = str(row.plate_text).strip()
            encoded = [char_map[c] for c in text if c in char_map and char_map[c] != 0]
            if not encoded:
                continue  # nothing left after vocab filtering -> unusable for CTC
            self.samples.append((row.image_path, encoded, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, encoded, text = self.samples[idx]
        try:
            img = Image.open(self.root / str(rel_path)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 64))
        if self.transform:
            img = self.transform(img)
        return img, torch.LongTensor(encoded), len(encoded), text, str(rel_path), "real"


def ocr_collate(batch):
    imgs, targets, target_lens, texts, paths, types = zip(*batch)
    imgs = torch.stack(imgs)
    targets = torch.cat(targets)
    target_lens = torch.LongTensor(target_lens)
    return imgs, targets, target_lens, texts, paths, types


def _load_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).fillna("")
    return pd.DataFrame({
        "image_path": df["image_path"].astype(str),
        "plate_text": df["plate_text"].astype(str),
    })


def _filter_existing(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    keep = [(str(r.plate_text).strip() != "") and (root / str(r.image_path)).exists()
            for r in df.itertuples(index=False)]
    return df[keep]


def train_lao_ocr(epochs=60, batch_size=32, lr=1e-3, data_dir=None, save_name=None, tag=None, patience=None):
    print("\n=== Lao OCR Fallback Trainer (ResNetCRNN + CTC) ===")
    print(f"Device: {DEVICE}")

    root = Path(data_dir) if data_dir is not None else DATA_DIR

    if save_name is not None:
        save_path = WEIGHTS_DIR / save_name
    elif tag is not None:
        save_path = WEIGHTS_DIR / f"ocr_model_lao_{tag}.pth"
    else:
        save_path = DEFAULT_SAVE_PATH

    if not CHAR_MAP_PATH.exists():
        raise FileNotFoundError(f"Lao char map not found: {CHAR_MAP_PATH}")
    with open(CHAR_MAP_PATH, "r", encoding="utf-8") as f:
        int_to_char = json.load(f)
    char_to_int = {v: int(k) for k, v in int_to_char.items()}
    print(f"Vocab size (incl. blank): {len(int_to_char)}")
    print(f"Dataset  : {root}")
    print(f"Weights  : {save_path}")

    train_csv = root / "ground_truth_train.csv"
    val_csv = root / "ground_truth_validation.csv"
    for p in (train_csv, val_csv):
        if not p.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {p}")

    train_df = _filter_existing(_load_split(train_csv), root)
    val_df = _filter_existing(_load_split(val_csv), root)
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    train_ds = LaoOCRDataset(train_df, root, char_map=char_to_int, transform=get_ocr_transforms(True))
    val_ds = LaoOCRDataset(val_df, root, char_map=char_to_int, transform=get_ocr_transforms(False))
    print(f"Usable for CTC -> Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=ocr_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=ocr_collate, num_workers=0)

    model = ResNetCRNN(1, len(int_to_char), hidden_size=256).to(DEVICE)

    # NOTE: intentionally does NOT resume from any checkpoint — a fresh run
    # learns entirely under the train/serve-matched preprocessing.
    if save_path.exists():
        print(f"⚠️  {save_path.name} exists — will be OVERWRITTEN on first improvement.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_score = 100.0
    best_cer = 1.0
    patience = 0
    early_stop_patience = patience if patience else 12

    def validate():
        model.eval()
        cer_sum, tot, valid_fmt = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, _, _, texts, _, _ = batch
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                preds = best_path_decode(out.softmax(-1), int_to_char)
                for i, gt in enumerate(texts):
                    pred_fmt = format_lao_plate(preds[i])
                    if is_valid_lao_plate(pred_fmt):
                        valid_fmt += 1
                    cer_sum += min(1.0, levenshtein(pred_fmt, gt) / max(1, len(gt)))
                    tot += 1
        avg_cer = cer_sum / max(1, tot)
        fmt_acc = valid_fmt / max(1, tot)
        print(f"       [Val] Format-Valid: {fmt_acc:.2%} ({valid_fmt}/{tot}), CER: {avg_cer:.4f}")
        return avg_cer, fmt_acc

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"LaoOCR Ep {ep + 1}/{epochs}")
        for batch in pbar:
            imgs, tg, tg_lens, _, _, _ = batch
            imgs, tg, tg_lens = imgs.to(DEVICE), tg.to(DEVICE), tg_lens.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            logp = out.log_softmax(-1).permute(1, 0, 2)
            input_lengths = torch.full((imgs.size(0),), out.size(1), dtype=torch.long).to(DEVICE)
            loss = criterion(logp, tg, input_lengths, tg_lens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_cer, val_fmt = validate()
        combined = val_cer + (1.0 - val_fmt)
        scheduler.step(combined)

        if combined < best_score:
            best_score = combined
            best_cer = val_cer
            patience = 0
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "int_to_char": int_to_char,
                "epoch": ep,
                "cer": val_cer,
                "fmt_acc": val_fmt,
                "score": combined,
                "lang": "lao",
            }, save_path)
            print(f"   └── Score {combined:.4f} (CER {val_cer:.4f}) -> New Best! Saved {save_path.name}")
        else:
            patience += 1
            print(f"   └── Score {combined:.4f} (Best {best_score:.4f}) [Patience {patience}/{early_stop_patience}]")
            if patience >= early_stop_patience:
                print("Early stopping.")
                break

    print("\n" + "=" * 60)
    print(f"🏁 Lao OCR Training Complete — Best CER: {best_cer:.4f}, Score: {best_score:.4f}")
    print(f"   Weights: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lao OCR fallback (ResNetCRNN + CTC)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root containing ground_truth_train/validation.csv (default: datasets/Lao/lao-plate-dataset)")
    parser.add_argument("--save-name", type=str, default=None,
                        help="Custom filename in weights/ (default: ocr_model_lao.pth)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag suffix (e.g. v1 -> ocr_model_lao_v1.pth)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience in epochs (default: 12)")
    args = parser.parse_args()

    train_lao_ocr(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        data_dir=args.data_dir,
        save_name=args.save_name,
        tag=args.tag,
        patience=args.patience,
    )
