"""
src/preprocess_registry.py

Single Source of Truth for per-model preprocessing / post-processing specs.

WHY THIS EXISTS
===============
Every detection model in this project was fine-tuned with a DIFFERENT geometry
convention, and the production inference path silently drifted away from the
training convention. That drift — not "padding vs no padding" — is the root
cause of the accuracy gaps observed between candidates:

  ┌────────────────────────────┬───────────────┬─────────────────────────────┐
  │ Model (training script)    │ Train resize  │ Train aspect (measured)     │
  ├────────────────────────────┼───────────────┼─────────────────────────────┤
  │ M1 full-scene (all)        │ STRETCH       │ ~4:3  (med 1.33)            │
  │ M2 components (D-FINE)     │ STRETCH       │ ~2:1  (med 1.95)            │
  │ M3A charbox  (RF-DETR)     │ STRETCH       │ ~3.7:1 (med 3.73)           │
  └────────────────────────────┴───────────────┴─────────────────────────────┘

  - RF-DETR (rfdetr 1.10.1): train & predict = pure STRETCH to a square
    resolution (Base 560 / Small 512 / Nano 384) + ImageNet mean/std.
    NO letterbox, NO padding. (detr.py: F.resize(t, [res, res])).
  - D-FINE / PicoDet via LibreYOLO: train & predict = STRETCH to 640x640
    (D-FINE) or 416x416 (PicoDet). Confirmed by LibreYOLO's own DeepStream
    export config: D-FINE maintain-aspect-ratio=0 (stretch, no letterbox).
  - Ultralytics YOLO / RT-DETR: classic LETTERBOX (keep aspect + grey pad).
  - ONNX exports pin the resolution: D-FINE 640, RF-DETR-Base 560,
    RF-DETR-Small 512, PicoDet 416.

RULES
=====
1.  The tensor fed to the network is ALWAYS square (1:1). Aspect ratio is
    handled by the RESIZE MODE, not by padding to a different shape:
      - STRETCH   : keep the model's native aspect (image is pre-cropped to
                    the same aspect it was trained on), then stretch.
      - LETTERBOX : keep the image's aspect, pad with grey (YOLO family).
2.  Never letterbox a model that was trained with stretch (RF-DETR, D-FINE,
    PicoDet) and never stretch a model trained with letterbox (YOLO).
3.  Serve the input at the TRAINING pixel geometry (aspect * tensor size),
    not at whatever the previous pipeline stage happened to emit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ResizeMode:
    STRETCH = "stretch"          # ignore aspect: fill the tensor (RF-DETR / D-FINE / PicoDet)
    LETTERBOX = "letterbox"      # keep aspect, grey pad (Ultralytics YOLO / RT-DETR)
    ASPECT_STRETCH = "aspect_stretch"  # resize W and H independently to fixed WxH (non-square)


class Normalization:
    """Normalisation schemes used by the backbones in this repo."""
    IMAGENET = "imagenet"        # mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]  (RF-DETR)
    UNIT = "unit"                # x / 255                                        (D-FINE, PicoDet, YOLO)


@dataclass(frozen=True)
class ModelPreprocessSpec:
    """Immutable preprocessing/post-processing contract for one detector."""
    name: str
    resize_mode: str                     # ResizeMode.*
    tensor_h: int                        # network input height (square unless ASPECT_STRETCH)
    tensor_w: int                        # network input width
    normalization: str                   # Normalization.*
    train_aspect_w_over_h: float = 1.0   # median train-image aspect (W/H); informational
    # how detections must be mapped back to source pixels
    postprocess: str = "identity"        # identity | divide_by_upscale_scale
    notes: str = ""

    @property
    def is_square(self) -> bool:
        return self.tensor_w == self.tensor_h

    def recommended_input_shape(self) -> tuple[int, int]:
        """Pixel shape (w, h) the caller should provide BEFORE the final resize.

        For stretch-trained models we keep the TRAIN aspect so the effective
        object scale matches training (this is the whole point of the fix):
        e.g. M2 trained on ~2:1 crops at 640 tensor -> feed 1280x640 pixels,
        M3A trained on ~3.7:1 crops at 560 tensor -> feed 2088x560 pixels.
        """
        a = self.train_aspect_w_over_h
        def _even_floor(x: float) -> int:
            v = int(x)
            return v - (v % 2)  # even alignment, deterministic
        if a >= 1.0:
            w = min(_even_floor(self.tensor_w * a), self.MAX_INPUT_W)
            return (w, self.tensor_h)
        h = min(_even_floor(self.tensor_h / a), self.MAX_INPUT_W)
        return (self.tensor_w, h)

    # Hard ceiling so upstream callers cannot allocate absurd buffers.
    # 2560 is still trivial memory-wise (the network tensor stays square).
    MAX_INPUT_W = 2560


# ---------------------------------------------------------------------------
# Model 1 — full-scene plate detectors
# ---------------------------------------------------------------------------
M1_RFDETR_BASE = ModelPreprocessSpec(
    name="plate_detector_rfdetr (RF-DETR-Base)",
    resize_mode=ResizeMode.STRETCH,
    tensor_h=560, tensor_w=560,
    normalization=Normalization.IMAGENET,
    train_aspect_w_over_h=4 / 3,
    postprocess="identity",
    notes="Native resolution 560x560 (ONNX-verified). Do NOT letterbox. Do NOT "
          "pre-shrink to 640 before predict(); pass the frame and let rfdetr "
          "stretch it to 560. Boxes come back on the source pixel grid.",
)

M1_RFDETR_SMALL = ModelPreprocessSpec(
    name="plate_detector_rfdetr_small (RF-DETR-Small)",
    resize_mode=ResizeMode.STRETCH,
    tensor_h=512, tensor_w=512,
    normalization=Normalization.IMAGENET,
    train_aspect_w_over_h=4 / 3,
    postprocess="identity",
    notes="Native resolution 512x512 (ONNX-verified).",
)

M1_DFINE = ModelPreprocessSpec(
    name="plate_detector_dfine_* (D-FINE Nano/Small via LibreYOLO)",
    resize_mode=ResizeMode.STRETCH,
    tensor_h=640, tensor_w=640,
    normalization=Normalization.UNIT,
    train_aspect_w_over_h=4 / 3,
    postprocess="identity",
    notes="LibreYOLO trains/predicts with pure stretch (maintain-aspect-ratio=0 "
          "in its own DeepStream config). imgsz must stay 640 at predict time.",
)

M1_PICODET = ModelPreprocessSpec(
    name="plate_detector_picodet_* (PicoDet-S/M via LibreYOLO)",
    resize_mode=ResizeMode.STRETCH,
    tensor_h=416, tensor_w=416,
    normalization=Normalization.UNIT,
    train_aspect_w_over_h=4 / 3,
    postprocess="identity",
    notes="CRITICAL: trained at imgsz=416 but the wrapper previously fed 640. "
          "Always pass imgsz=416 at predict/export time (ONNX input is 416x416).",
)

M1_YOLO_LETTERBOX = ModelPreprocessSpec(
    name="yolo11* / rtdetr (Ultralytics)",
    resize_mode=ResizeMode.LETTERBOX,
    tensor_h=640, tensor_w=640,
    normalization=Normalization.UNIT,
    train_aspect_w_over_h=4 / 3,
    postprocess="identity",
    notes="Ultralytics handles letterboxing internally; pass imgsz only.",
)

# ---------------------------------------------------------------------------
# Model 2 — component detector (plate_char / province) on rectified plates.
# Trained on Roboflow rectified crops, measured median aspect 1.95 (~2:1),
# fine-tuned with D-FINE at 640x640 stretch.
# The old pipeline fed the 320x160 rectified plate directly -> objects appeared
# ~3.5x smaller in tensor space than during training -> recall loss.
# Fix: upscale to the TRAIN pixel geometry 1280x640 (2:1) before inference.
# ---------------------------------------------------------------------------
M2_COMPONENTS = ModelPreprocessSpec(
    name="component_detector_dfine_nano (D-FINE Nano)",
    resize_mode=ResizeMode.STRETCH,
    tensor_h=640, tensor_w=640,
    normalization=Normalization.UNIT,
    train_aspect_w_over_h=2.0,
    postprocess="divide_by_upscale_scale",
    notes="Feed rectified plates upscaled to 1280x640 (train aspect 2:1) and "
          "divide returned xyxy by the upscale factor. Do NOT letterbox.",
)

# ---------------------------------------------------------------------------
# Model 3A — character box detector.
# Trained on Roboflow char crops, measured median aspect 3.73 (~3.7:1),
# fine-tuned with RF-DETR-Base (560x560 stretch + ImageNet norm).
# Keep the train pixel geometry: 560 * 3.73 ~= 2088 wide (capped), height 560.
# Also: NO CLAHE at inference — it never existed in training.
# ---------------------------------------------------------------------------
M3A_CHARBOX = ModelPreprocessSpec(
    name="character_box_detector_rfdetr (RF-DETR-Base)",
    resize_mode=ResizeMode.STRETCH,
    tensor_h=560, tensor_w=560,
    normalization=Normalization.IMAGENET,
    train_aspect_w_over_h=3.73,
    postprocess="divide_by_upscale_scale",
    notes="Feed char crops upscaled to train aspect (~3.7:1, height 560) and "
          "divide returned xyxy by the upscale factor. No CLAHE / unsharp at "
          "inference (domain mismatch with training data).",
)

M3A_CHARBOX_DFINE = ModelPreprocessSpec(
    name="character_box_detector_dfine_* (D-FINE Nano/Small)",
    resize_mode=ResizeMode.STRETCH,
    tensor_h=640, tensor_w=640,
    normalization=Normalization.UNIT,
    train_aspect_w_over_h=3.73,
    postprocess="divide_by_upscale_scale",
    notes="Same geometry rule as the RF-DETR variant but 640 tensor + unit norm.",
)


def get_m1_spec(model_filename: str) -> ModelPreprocessSpec:
    """Pick the Model 1 spec from the active weight filename."""
    n = model_filename.lower()
    if "picodet" in n:
        return M1_PICODET
    if "rfdetr" in n and "obb" not in n:
        return M1_RFDETR_SMALL if "small" in n else M1_RFDETR_BASE
    if "dfine" in n or "libre" in n or "rtdetrv2" in n:
        return M1_DFINE
    if "rtdetr" in n:
        return M1_YOLO_LETTERBOX
    if "yolo" in n:
        return M1_YOLO_LETTERBOX
    # Unknown checkpoint: assume the permissive-stack default (D-FINE geometry).
    return M1_DFINE


def get_m2_spec(model_filename: str) -> ModelPreprocessSpec:
    n = model_filename.lower()
    if "rfdetr" in n:
        return M1_RFDETR_BASE  # same family geometry (560 stretch + ImageNet)
    return M2_COMPONENTS


def get_m3a_spec(model_filename: str) -> ModelPreprocessSpec:
    n = model_filename.lower()
    if "dfine" in n:
        return M3A_CHARBOX_DFINE
    if "rtdetr" in n or "yolo" in n:
        return M1_YOLO_LETTERBOX
    return M3A_CHARBOX


def stretch_resize(img, spec: ModelPreprocessSpec):
    """Resize `img` (HxWx3) to the spec tensor with PURE STRETCH (no padding).

    Matches exactly what rfdetr's predict() and LibreYOLO's pipeline do
    (bilinear / INTER_LINEAR, no antialias, no letterbox).
    """
    import cv2

    h, w = img.shape[:2]
    if (w, h) == (spec.tensor_w, spec.tensor_h):
        return img
    return cv2.resize(img, (spec.tensor_w, spec.tensor_h), interpolation=cv2.INTER_LINEAR)


def upscale_to_train_geometry(img, spec: ModelPreprocessSpec):
    """Upscale a crop to the TRAIN pixel geometry before the final stretch.

    The crop keeps its own aspect (it comes from a physical plate); we resize
    it to the train pixel geometry so that after the model's internal stretch
    to the square tensor, the effective object scale matches training.

    Returns (resized_img, (scale_x, scale_y)). Post-process must divide
    returned xyxy by these PER-AXIS scales to get source-pixel coordinates
    (scale_x == scale_y when the crop aspect equals the train aspect).
    """
    import cv2

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img, (1.0, 1.0)
    target_w, target_h = spec.recommended_input_shape()
    scale_x = target_w / float(w)
    scale_y = target_h / float(h)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return resized, (scale_x, scale_y)


def divide_boxes_by_scale(xyxy, scale):
    """Map xyxy boxes (N,4) from the upscaled grid back to source pixels."""
    import numpy as np

    sx, sy = scale
    a = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0:
        return a
    out = a.copy()
    out[:, [0, 2]] /= sx
    out[:, [1, 3]] /= sy
    return out
