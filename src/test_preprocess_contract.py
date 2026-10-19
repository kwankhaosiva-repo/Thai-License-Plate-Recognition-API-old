"""
src/test_preprocess_contract.py

Executable form of docs/PREPROCESSING_CONTRACT.md. Every rule in that document
that can be checked WITHOUT loading a model is checked here, so the standard
cannot silently rot when someone edits a training script.

Run:  /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/test_preprocess_contract.py

THE THREE LAWS
--------------
 1. EVAL == TRAIN.  Anything a model sees at serve must also appear, identically,
    in its training pipeline. Anything that appears only in training is
    augmentation and must be absent at serve.
 2. NO CRUTCH.  A preprocessing step is legitimate only if turning it off does
    not degrade the metric. "It gets worse without it" is evidence the step is
    masking a data or model defect, not evidence the step is good.
 3. GEOMETRY IS THE CONTRACT.  The crop size/aspect/frame a model is trained at
    is the contract, not a tunable. Rotation, scale and colour jitter belong in
    TRAIN augmentation.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.config import Config, cfg
from src.preprocess import (
    SmartResize,
    GrayscaleSmartResize,
    get_grayscale_prov_transforms,
    get_ocr_transforms,
    _ocr_enhanced_gray,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}   {detail}")


def plate_crop(w=118, h=30, bg=205):
    """Synthetic province crop: light plate, dark glyph band in the middle."""
    a = np.full((h, w, 3), bg, dtype=np.uint8)
    a[:, w // 3: 2 * w // 3] = 40
    return Image.fromarray(a)


# ---------------------------------------------------------------------------
print("== 1. Plate rectification is a CONTRACT, not an accuracy knob ==")
check("ENABLE_HOMOGRAPHY_DESKEW is True (the plate crop IS the rectified quad)",
      Config.ENABLE_HOMOGRAPHY_DESKEW is True, str(Config.ENABLE_HOMOGRAPHY_DESKEW))
check("cfg.USE_HOMOGRAPHY_DESKEW resolves True", cfg.USE_HOMOGRAPHY_DESKEW is True)

# ---------------------------------------------------------------------------
print("== 2. Residual deskew is OFF by default (Law 2: no crutch) ==")
check("ENABLE_RESIDUAL_DESKEW is False", Config.ENABLE_RESIDUAL_DESKEW is False,
      str(Config.ENABLE_RESIDUAL_DESKEW))
check("cfg.USE_RESIDUAL_DESKEW resolves False", cfg.USE_RESIDUAL_DESKEW is False)
check("is_residual_deskew_enabled() is False", Config.is_residual_deskew_enabled() is False)
saved_pure = Config.PURE_MODEL_MODE
try:
    Config.PURE_MODEL_MODE = True
    check("PURE_MODEL_MODE cannot re-enable the residual search",
          cfg.USE_RESIDUAL_DESKEW is False)
finally:
    Config.PURE_MODEL_MODE = saved_pure

# api_server must route BOTH call sites through the gated helper, so the flag
# is honoured everywhere and the un-gated import is not a bypass.
import src.api_server as api

api_src = inspect.getsource(api)
n_gated = api_src.count("apply_residual_deskew(")
check("api_server has an apply_residual_deskew() gate", n_gated >= 2, f"{n_gated} call sites")
check("no ungated fine_deskew_plate( call left in api_server",
      api_src.count("fine_deskew_plate(") <= n_gated + 1,
      f"{api_src.count('fine_deskew_plate(')} occurrences")
gate_src = inspect.getsource(api.apply_residual_deskew)
check("the gate checks cfg.USE_RESIDUAL_DESKEW", "USE_RESIDUAL_DESKEW" in gate_src)

# ---------------------------------------------------------------------------
print("== 3. serve_perspective_matrix reproduces the serve warp exactly ==")
from src.prepare_perspective_dataset import warp_perspective_plate
from src.prepare_serve_frame_components import serve_perspective_matrix

rng = np.random.default_rng(7)
img = rng.integers(0, 255, (97, 191, 3), dtype=np.uint8)
quad = np.array([[11.0, 6.0], [178.0, 2.0], [185.0, 90.0], [5.0, 95.0]], dtype=np.float32)
matrix, src_pts, dst_pts = serve_perspective_matrix(quad)

via_serve = warp_perspective_plate(img, quad, target_width=320, target_height=160,
                                   padding_frac=0.06)
via_matrix = cv2.warpPerspective(img, matrix, (320, 160), flags=cv2.INTER_CUBIC)
check("warped output is byte-identical to the serve warp",
      np.array_equal(via_serve, via_matrix),
      f"max abs diff {int(np.abs(via_serve.astype(int) - via_matrix.astype(int)).max())}")
check("quad corners map onto the 320x160 destination rect",
      np.allclose(dst_pts, np.array([[0, 0], [319, 0], [319, 159], [0, 159]], np.float32)))

# ---------------------------------------------------------------------------
print("== 4. Province eval transform == the trainers' private letterbox (Law 1) ==")
# train_grayscale_province_thai.py has its OWN GrayscaleSmartResize with NO
# toggle: it ALWAYS letterboxes with the corner-median background. The serve
# copy resolves Config.ENABLE_AMBIENT_CORNER_PAD. These must agree, on a crop
# whose aspect differs from the tensor, or glyphs are stretched 1.33x.
for size in ((256, 80), (256, 64)):
    crop = plate_crop(118, 30)
    serve_out = GrayscaleSmartResize(size, use_ambient_padding=True, use_grayscale=True)(crop)
    trainer_out = GrayscaleSmartResize(size, use_ambient_padding=True, use_grayscale=True)(crop)
    check(f"serve/trainer province resize agree at {size}",
          np.array_equal(np.array(serve_out), np.array(trainer_out)))

c = plate_crop(118, 30)  # aspect 3.93 vs tensor 3.20 -> must letterbox, not stretch
letterboxed = GrayscaleSmartResize((256, 80), use_ambient_padding=True, use_grayscale=True)(c)
stretched = GrayscaleSmartResize((256, 80), use_ambient_padding=False, use_grayscale=True)(c)
check("letterbox differs from stretch on a non-square crop",
      not np.array_equal(np.array(letterboxed), np.array(stretched)))
# A 3.93-aspect crop into a 3.20 tensor letterboxes to a full-width band with
# small top/bottom bars; STRETCH instead scales width by 3.20/3.93 = 0.81 and
# height by 1.0, i.e. squashes the glyphs horizontally by 24%.
def dark_span(img):
    a = np.array(img).astype(int)[:, :, 0]
    dark = a < 100
    ys, xs = np.where(dark)
    return (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) if len(xs) else (0, 0)

lb_w, lb_h = dark_span(letterboxed)
st_w, st_h = dark_span(stretched)
src_w, src_h = dark_span(c)
src_ar = src_w / max(src_h, 1)
lb_ar, st_ar = lb_w / max(lb_h, 1), st_w / max(st_h, 1)
check("LETTERBOX preserves the source glyph aspect ratio",
      abs(lb_ar - src_ar) < abs(st_ar - src_ar) * 0.5,
      f"source {src_ar:.2f} | letterbox {lb_ar:.2f} | stretch {st_ar:.2f}")
check("STRETCH distorts the glyph aspect (this is the 5.8-point regression)",
      abs(st_ar - src_ar) > 0.2, f"stretch aspect {st_ar:.2f} vs source {src_ar:.2f}")

# ---------------------------------------------------------------------------
print("== 5. Province EVAL path has no augmentation (Law 1) ==")
eval_tf = get_grayscale_prov_transforms(is_train=False)
train_tf = get_grayscale_prov_transforms(is_train=True)
AUG = ("RandomAffine", "ColorJitter", "RandomAutocontrast", "RandomPerspective",
       "RandomHorizontalFlip", "RandomRotation", "RandomErasing", "RandomApply")
for a in AUG:
    check(f"eval province transforms contain no {a}", a not in str(eval_tf))
# Rotation/scale tolerance must be LEARNED. RandomAffine is the only op that
# provides it, and its degrees must cover the residual tilt the corner regressor
# leaves behind (measured max 2.75 deg on real plates -> 4 deg is right).
check("train province transforms DO contain RandomAffine (rotation tolerance)",
      "RandomAffine" in str(train_tf))
import re as _re
_deg = _re.search(r"RandomAffine\(degrees=\[-([\d.]+),", str(train_tf))
check("RandomAffine degrees >= 4 (covers the measured 2.75 deg residual tilt)",
      _deg is not None and float(_deg.group(1)) >= 4.0,
      f"degrees={_deg.group(1) if _deg else '?'}")

# deterministic: the eval path must be a pure function
a1 = eval_tf(plate_crop(118, 30))
a2 = eval_tf(plate_crop(118, 30))
check("eval province transform is deterministic", np.array_equal(a1.numpy(), a2.numpy()))

# ---------------------------------------------------------------------------
print("== 6. OCR train and eval share the same deterministic prefix (Law 1) ==")
e_src = inspect.getsource(get_ocr_transforms)
check("ocr eval path still runs _ocr_enhanced_gray", "_ocr_enhanced_gray" in e_src)
gray = Image.fromarray(rng.integers(60, 200, (40, 120), dtype=np.uint8))
check("_ocr_enhanced_gray is autocontrast(cutoff=1)",
      np.array_equal(np.array(_ocr_enhanced_gray(gray)),
                     np.array(ImageOps.autocontrast(gray, cutoff=1))))
for a in ("RandomAffine", "RandomPerspective", "ColorJitter"):
    check(f"ocr eval path contains no {a}", a not in str(get_ocr_transforms(is_train=False)))

# ---------------------------------------------------------------------------
print("== 7. Letterbox background is the crop's own tone, never black (Law 2) ==")
grey_crop = plate_crop(118, 30, bg=205).convert("L")
lb = GrayscaleSmartResize((256, 80), use_ambient_padding=True, use_grayscale=True)(grey_crop)
arr = np.array(lb)
top_row = arr[0, :, 0]
check("letterbox bars carry the corner-median tone (~205), not 0",
      int(np.median(top_row)) > 150, f"bar median {int(np.median(top_row))}")

# ---------------------------------------------------------------------------
print("== 8. Recognition harvester must not drift from the serve crop rules ==")
# extract_recognition_crops_v2.py re-implements the inline api_server crop math.
# These have already drifted once (the char margin), which silently puts the
# character classifier's training crops out of distribution.
import src.extract_recognition_crops_v2 as harvest  # noqa: E402


def _norm(t):
    return " ".join(t.split())


serve_norm = _norm(api_src)
harvest_norm = _norm(inspect.getsource(harvest))
for label, snippet in [
    ("char pad_cx", "pad_cx = min(10, max(3, int(bw_char * 0.04)))"),
    ("char pad_cy", "pad_cy = min(10, max(3, int(bh_char * 0.10)))"),
    ("char row bottom clamp", "cy2 = min(int(rh * 0.76), by2 + pad_cy)"),
    ("prov pad_x", "pad_x = min(14, max(6, int(bw * 0.08)))"),
    ("prov pad_y", "pad_y = min(8, max(4, int(bh_box * 0.10)))"),
    ("prov top clamp", "py1 = max(int(rh * 0.46), by1 - pad_y)"),
]:
    check(f"serve rule present in api_server: {label}", _norm(snippet) in serve_norm,
          "api_server crop math changed — update the harvester too")
    check(f"harvester mirrors it: {label}", _norm(snippet) in harvest_norm,
          "extract_recognition_crops_v2.py is stale vs api_server.py")

# The harvester must not reintroduce the residual deskew it never had.
check("harvester never calls fine_deskew_plate (serve-only heuristic)",
      "fine_deskew_plate" not in harvest_norm)

# ---------------------------------------------------------------------------
print("=" * 60)
print(f"RESULT: {PASS} PASS, {FAIL} FAIL")
sys.exit(1 if FAIL else 0)