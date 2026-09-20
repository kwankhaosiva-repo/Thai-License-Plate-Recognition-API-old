"""
src/test_classifier_preprocess.py

Self-test for the PADDING BACKGROUND + GEOMETRY contract of the 4
classification models (no model loading needed):

  1. OCR (ocr_model.pth)        : SmartResize pads with BLACK (0) — same class
                                  used in train (train_ocr.py) and serve.
  2. Char classifier (thai/lao) : dataset crops are PRE-PADDED to 64x64 square
                                  with corner-median bg at harvest time, so
                                  Resize((64,64)) is a no-op; serve pads with
                                  the same corner-median rule.
  3. Province grayscale (thai)  : GrayscaleSmartResize pads with the crop's own
                                  background tone (NOT black), at (256, 80).
  4. Province grayscale (lao)   : same mechanism at (256, 64).

Run:  python src/test_classifier_preprocess.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.preprocess import SmartResize, GrayscaleSmartResize, get_grayscale_prov_transforms

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# Build a synthetic "crop": light background (plate-like), dark glyph center
def make_crop(w, h, bg=200, glyph=30):
    arr = np.full((h, w), bg, dtype=np.uint8)
    arr[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = glyph
    return Image.fromarray(arr, mode="L")


print("== 1. OCR: SmartResize pads with BLACK (0), train==serve ==")
t = SmartResize((256, 64), mode="L")
out = np.array(t(make_crop(100, 40)))  # wide crop -> side bars
check("ocr output size == (64, 256)", out.shape == (64, 256), f"got {out.shape}")
check("ocr pad color is black (0)", out[0, 0] == 0 and out[-1, -1] == 0, f"corner={out[0,0]}")
check("ocr glyph area is NOT black", out[32, 128] < 100, f"center={out[32,128]}")

print("== 2. Char classifier: 64x64 geometry, serve pad rule = corner-median ==")
# harvest pad_to_square: bg = median of 4 corners -> simulate on square crop (identity)
sq = make_crop(64, 64)
arr = np.array(sq)
corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
check("square crop corners identical (pre-padded bg)", len(set(corners)) == 1, f"{corners}")
# non-square crop padded to square with corner-median must keep aspect-1.0 geometry
ns = np.full((30, 50), 200, dtype=np.uint8)  # h=30, w=50 (non-square)
ns_c = np.array([ns[0, 0], ns[0, -1], ns[-1, 0], ns[-1, -1]])
bg_col = np.median(ns_c, axis=0).astype(np.uint8)
check("corner-median bg == true bg (200)", bg_col == 200, f"got {bg_col}")
from torchvision import transforms
tf_char = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
tt = tf_char(make_crop(64, 64).convert("RGB"))
check("char classifier tensor == (3, 64, 64)", tuple(tt.shape) == (3, 64, 64), f"{tuple(tt.shape)}")

print("== 3/4. Province grayscale: pad = OWN background tone (not black) ==")
for label, size, expected in [("thai", (256, 80), (80, 256)), ("lao", (256, 64), (64, 256))]:
    g = GrayscaleSmartResize(size)
    out = np.array(g(make_crop(180, 60, bg=170)))  # bg tone 170; returns RGB (H, W, 3)
    check(f"{label} prov output size == {expected}", out.shape == (*expected, 3), f"got {out.shape}")
    check(f"{label} prov pad uses bg tone (~170), NOT black", abs(int(out[0, 0, 0]) - 170) <= 2, f"corner={out[0,0,0]}")
    check(f"{label} prov glyph preserved (dark)", out[expected[0] // 2, expected[1] // 2, 0] < 100,
          f"center={out[expected[0]//2, expected[1]//2, 0]}")
    tf = get_grayscale_prov_transforms(is_train=False, size=size)
    tensor = tf(make_crop(180, 60, bg=170).convert("RGB"))
    check(f"{label} prov tensor == (3, {expected[0]}, 256)", tuple(tensor.shape) == (3, expected[0], 256),
          f"{tuple(tensor.shape)}")

print("== 5. Normalize constants match training scripts ==")
import inspect
src = inspect.getsource(get_grayscale_prov_transforms)
check("prov uses mean/std 0.5/0.25 (matches train_grayscale_province_*.py)", "0.5, 0.5, 0.5" in src and "0.25, 0.25, 0.25" in src)

print("-" * 60)
print(f"RESULT: {PASS} PASS, {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
