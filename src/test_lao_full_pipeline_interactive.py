#!/usr/bin/env python3
"""
Interactive End-to-End Test Tool for Lao & Thai License Plate Recognition Pipeline.
Tests raw scene images through the complete multi-stage workflow:
  1. Model 1   : Plate Detection (PicoDet-S)
  2. Homography: 4-Corner Regressor & Perspective Rectification (320x160)
  3. Model 1.5 : Country Classification (MobileNetV3-Small: Thai vs Laos)
  4. Country Branches:
     - Thai: Model 2 Layout (Thai) -> Thai Char Box & Clf -> Thai Province Model
     - Lao : 3:1 Padding -> Model 3A Lao Char Detector -> Lao MobileNetV4 Recognizer -> Lao Province Detector/Clf
  5. Visualization: Full Dashboard with Scene, Rectified Plate, Yellow Province Box & Green Character Boxes

Usage:
  python src/test_lao_full_pipeline_interactive.py "<folder_or_image_path>"
  python src/test_lao_full_pipeline_interactive.py "datasets/Lao/Lao License Plates.v2i.yolov11/test/images"
  python src/test_lao_full_pipeline_interactive.py "datasets/Lao/Lao License Plates.v2i.yolov11/test/images/-8093_jpeg.rf.8c17b31e9e7c93b424b715de3249e9a1.jpg"

Controls:
  [SPACE]    : Next image
  [B]        : Previous image
  [S]        : Save current visualization to disk
  [Q] / [ESC]: Quit
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Ensure project root is in sys.path when running script directly from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.ops import nms
from PIL import Image, ImageOps, ImageDraw, ImageFont
import onnxruntime as ort

from libreyolo import LibreYOLO
from src.config import cfg
from src.models import ResNetProvinceClassifier, DigitClassifier
from src.prepare_perspective_dataset import warp_perspective_plate
from src.preprocess import apply_residual_deskew, get_grayscale_prov_transforms
from src.validators import (
    format_lao_plate,
    is_valid_lao_plate,
    format_thai_plate,
    is_valid_plate as is_valid_thai_plate,
)


# -------------------------------------------------------------------------
# Lao & Thai Font Support for OpenCV / PIL rendering
# -------------------------------------------------------------------------
_FONT_CACHE: Dict[int, Any] = {}

def get_font(size: int = 20) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "/System/Library/Fonts/Supplemental/Lao Sangam MN.ttf",
    ]
    font = None
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def draw_text_pil(
    img_bgr: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 20,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = None,
    padding: int = 4,
) -> np.ndarray:
    """Renders UTF-8 text (including Lao and Thai glyphs) onto an OpenCV BGR image using PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_size)

    x, y = pos
    bbox = draw.textbbox((x, y), text, font=font)
    if bg_color is not None:
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=bg_color,
        )
    # Outline for visibility
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# -------------------------------------------------------------------------
# Lao & Thai Multi-Stage Pipeline Engine
# -------------------------------------------------------------------------
class LaoFullPipelineTester:
    def __init__(self, device: str = "cpu", prov_mode: str = "2stage", use_onnx: bool = True):
        self.device = device
        self.prov_mode = prov_mode
        self.use_onnx = use_onnx
        self.weights_dir = cfg.WEIGHTS_DIR
        engine_str = "100% Pure ONNX Runtime" if use_onnx else "PyTorch & LibreYOLO"
        print("\n" + "=" * 70)
        print(f" Loading End-to-End Recognition Models ({engine_str}) ".center(70, "="))
        print("=" * 70)

        # 1. Model 1: Plate Detector (PicoDet-S v6pico320 / v6aug)
        self.m1_detector = None
        self.m1_onnx_sess = None
        m1_onnx_path = self.weights_dir / "plate_detector_picodet_s_v6pico320.onnx"
        m1_pt_path = self.weights_dir / "plate_detector_picodet_s_v6pico320.pt"

        if self.use_onnx and m1_onnx_path.exists():
            print(f"  [1/8] Model 1 (Plate Detector)  : {m1_onnx_path.name} (ONNX)")
            self.m1_onnx_sess = ort.InferenceSession(str(m1_onnx_path))
            self.m1_onnx_input = self.m1_onnx_sess.get_inputs()[0].name
        else:
            if not m1_pt_path.exists():
                m1_pt_path = self.weights_dir / "plate_detector_picodet_s_v6aug.pt"
            print(f"  [1/8] Model 1 (Plate Detector)  : {m1_pt_path.name}")
            self.m1_detector = LibreYOLO(str(m1_pt_path))

        # 2. Plate 4-Corner Homography Regressor (MobileNetV3-Small)
        corner_onnx_path = self.weights_dir / "plate_corner_regressor.onnx"
        corner_path = self.weights_dir / "plate_corner_regressor.pth"
        self.plate_corner_model = None
        self.corner_onnx_sess = None

        if self.use_onnx and corner_onnx_path.exists():
            print(f"  [2/8] Corner Regressor (Warp)   : {corner_onnx_path.name} (ONNX)")
            self.corner_onnx_sess = ort.InferenceSession(str(corner_onnx_path))
            self.corner_onnx_input = self.corner_onnx_sess.get_inputs()[0].name
        elif corner_path.exists():
            print(f"  [2/8] Corner Regressor (Warp)   : {corner_path.name}")
            base_c = models.mobilenet_v3_small()
            in_f = base_c.classifier[0].in_features
            base_c.classifier = nn.Sequential(
                nn.Linear(in_f, 128),
                nn.Hardswish(),
                nn.Dropout(0.1),
                nn.Linear(128, 8),
                nn.Sigmoid(),
            )
            ckpt_corner = torch.load(corner_path, map_location=self.device)
            base_c.load_state_dict(ckpt_corner.get("model_state", ckpt_corner))
            base_c.to(self.device).eval()
            self.plate_corner_model = base_c

        self.tf_corner = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 3. Model 1.5: Country Classifier (MobileNetV3-Small: Thai vs Laos)
        c_onnx_path = self.weights_dir / "country_classifier.onnx"
        c_path = self.weights_dir / "country_classifier.pth"
        self.country_model = None
        self.country_onnx_sess = None

        if self.use_onnx and c_onnx_path.exists():
            print(f"  [3/8] Model 1.5 (Country Clf)   : {c_onnx_path.name} (ONNX)")
            self.country_onnx_sess = ort.InferenceSession(str(c_onnx_path))
            self.country_onnx_input = self.country_onnx_sess.get_inputs()[0].name
        elif c_path.exists():
            print(f"  [3/8] Model 1.5 (Country Clf)   : {c_path.name}")
            self.country_model = models.mobilenet_v3_small(weights=None)
            in_f = self.country_model.classifier[3].in_features
            self.country_model.classifier[3] = nn.Linear(in_f, 2)
            ckpt_c = torch.load(c_path, map_location=self.device)
            self.country_model.load_state_dict(ckpt_c["model_state"])
            self.country_model.to(self.device).eval()

        self.tf_country = transforms.Compose([
            transforms.Resize((128, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 4. Model 2: Component Detector (PicoDet-S v6pico320 / v6frame)
        m2_path = self.weights_dir / "component_detector_picodet_s_v6pico320.pt"
        if not m2_path.exists():
            m2_path = self.weights_dir / "component_detector_picodet_s_v6frame.pt"
        if not m2_path.exists():
            m2_path = self.weights_dir / "component_detector_picodet_s_v2.pt"
        print(f"  [4/8] Model 2 (Component Det)  : {m2_path.name}")
        self.m2_detector = LibreYOLO(str(m2_path))

        # 5. Lao Model 3A: Character Box Detector (PicoDet-S Lao v6 / v5 / v4 ONNX)
        m3a_path = self.weights_dir / "character_box_detector_picodet_s_lao_m3a_v6.onnx"
        if not m3a_path.exists():
            m3a_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "character_box_detector_picodet_s_lao_m3a_v6.onnx"
        if not m3a_path.exists():
            m3a_path = self.weights_dir / "character_box_detector_picodet_s_lao_m3a_v5.onnx"
        if not m3a_path.exists():
            m3a_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "character_box_detector_picodet_s_lao_m3a_v5.onnx"
        if not m3a_path.exists():
            m3a_path = self.weights_dir / "character_box_detector_picodet_s_lao_m3a_v4.onnx"
        if not m3a_path.exists():
            m3a_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "character_box_detector_picodet_s_lao_m3a_v4.onnx"
        print(f"  [5/8] Lao M3A (Char Box Det)   : {m3a_path.name}")
        self.m3a_sess = ort.InferenceSession(str(m3a_path))
        self.m3a_input_name = self.m3a_sess.get_inputs()[0].name

        # 6. Lao Recognizer: MobileNetV4 Lao v7 / v6 ONNX
        rec_path = self.weights_dir / "character_recognition_mobilenetv4_lao_v7.onnx"
        if not rec_path.exists():
            rec_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "character_recognition_mobilenetv4_lao_v7.onnx"
        if not rec_path.exists():
            rec_path = self.weights_dir / "character_recognition_mobilenetv4_lao_v6.onnx"
        if not rec_path.exists():
            rec_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "character_recognition_mobilenetv4_lao_v6.onnx"

        map_path = self.weights_dir / "char_classifier_map_lao_v7.json"
        if not map_path.exists():
            map_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "char_classifier_map_lao_v7.json"
        if not map_path.exists():
            map_path = self.weights_dir / "char_classifier_map_lao_v6.json"
        if not map_path.exists():
            map_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "char_classifier_map_lao_v6.json"
        print(f"  [6/8] Lao Recognizer (MNv4)     : {rec_path.name}")
        self.rec_sess = ort.InferenceSession(str(rec_path))
        self.rec_input_name = self.rec_sess.get_inputs()[0].name
        with open(map_path, "r", encoding="utf-8") as f:
            self.char_map_lao = {int(k): v for k, v in json.load(f).items()}

        # 7. Lao Province Detector & Classifier
        prov_det_onnx_path = self.weights_dir / "province_detector_picodet_s_lao_v6.onnx"
        if not prov_det_onnx_path.exists():
            prov_det_onnx_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "province_detector_picodet_s_lao_v6.onnx"

        prov_det_path = self.weights_dir / "province_detector_picodet_s_lao_v6.pt"
        if not prov_det_path.exists():
            prov_det_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "province_detector_picodet_s_lao_v6.pt"
        if not prov_det_path.exists():
            prov_det_path = self.weights_dir / "province_detector_picodet_s_lao_v4.pt"

        self.lao_prov_detector = None
        self.lao_prov_det_onnx_sess = None
        if self.use_onnx and prov_det_onnx_path.exists():
            print(f"  [7/8] Lao Province Detector     : {prov_det_onnx_path.name} (ONNX)")
            self.lao_prov_det_onnx_sess = ort.InferenceSession(str(prov_det_onnx_path))
            self.lao_prov_det_onnx_input = self.lao_prov_det_onnx_sess.get_inputs()[0].name
        elif prov_det_path.exists():
            print(f"  [7/8] Lao Province Detector     : {prov_det_path.name}")
            try:
                self.lao_prov_detector = LibreYOLO(str(prov_det_path))
            except Exception as e:
                print(f"        Warning: Could not load Lao Province Detector: {e}")

        prov_map_path_lao = self.weights_dir / "province_map_lao.json"
        with open(prov_map_path_lao, "r", encoding="utf-8") as f:
            self.prov_map_lao = {int(k): v for k, v in json.load(f).items()}

        prov_onnx_path_lao = self.weights_dir / "province_model_grayscale_lao_v6.onnx"
        if not prov_onnx_path_lao.exists():
            prov_onnx_path_lao = cfg.PROJECT_ROOT / "chacrit" / "output model" / "province_model_grayscale_lao_v6.onnx"

        prov_path_lao = self.weights_dir / "province_model_grayscale_lao_v6.pth"
        if not prov_path_lao.exists():
            prov_path_lao = self.weights_dir / "province_model_grayscale_lao_v3.pth"

        self.prov_model_lao = None
        self.prov_model_lao_onnx_sess = None
        if self.use_onnx and prov_onnx_path_lao.exists():
            print(f"  [7/8b] Lao 2-Stage Prov Clf    : {prov_onnx_path_lao.name} ({len(self.prov_map_lao)} classes, ONNX)")
            self.prov_model_lao_onnx_sess = ort.InferenceSession(str(prov_onnx_path_lao))
            self.prov_model_lao_onnx_input = self.prov_model_lao_onnx_sess.get_inputs()[0].name
        elif prov_path_lao.exists():
            print(f"  [7/8b] Lao 2-Stage Prov Clf    : {prov_path_lao.name} ({len(self.prov_map_lao)} classes)")
            self.prov_model_lao = ResNetProvinceClassifier(n_classes=len(self.prov_map_lao), backbone="resnet18", pretrained=False)
            ckpt_p_lao = torch.load(prov_path_lao, map_location=self.device)
            self.prov_model_lao.load_state_dict(ckpt_p_lao.get("model_state", ckpt_p_lao))
            self.prov_model_lao.to(self.device).eval()

        # 8. Thai Models (Fallback for Thai plates)
        thai_charbox_path = self.weights_dir / "character_box_detector_picodet_s_v6pico320.pt"
        if not thai_charbox_path.exists():
            thai_charbox_path = self.weights_dir / "character_box_detector_picodet_s_v6frame.pt"
        if not thai_charbox_path.exists():
            thai_charbox_path = self.weights_dir / "character_box_detector_picodet_s_v2.pt"
        print(f"  [8/8] Thai Models (M3A/Prov)    : {thai_charbox_path.name}")
        self.thai_charbox_detector = LibreYOLO(str(thai_charbox_path)) if thai_charbox_path.exists() else None

        thai_cls_path = self.weights_dir / "character_classifier_v6pico320.pth"
        if not thai_cls_path.exists():
            thai_cls_path = self.weights_dir / "character_classifier_v6.pth"
        if not thai_cls_path.exists():
            thai_cls_path = self.weights_dir / "character_classifier.pth"
        thai_map_path = self.weights_dir / "char_classifier_map.json"
        with open(thai_map_path, "r", encoding="utf-8") as f:
            self.char_map_thai = {int(k): v for k, v in json.load(f).items()}
        from torchvision.models import mobilenet_v2
        self.thai_char_classifier = mobilenet_v2()
        self.thai_char_classifier.classifier[1] = nn.Linear(self.thai_char_classifier.classifier[1].in_features, len(self.char_map_thai))
        ckpt_tc = torch.load(thai_cls_path, map_location=self.device)
        self.thai_char_classifier.load_state_dict(ckpt_tc.get("model_state", ckpt_tc))
        self.thai_char_classifier.to(self.device).eval()

        # Thai Province Model
        prov_path_thai = self.weights_dir / "province_model_grayscale_thai_v6pico320.pth"
        if not prov_path_thai.exists():
            prov_path_thai = self.weights_dir / "province_model_grayscale_thai_v5.pth"
        if not prov_path_thai.exists():
            prov_path_thai = self.weights_dir / "province_model_grayscale_thai_v4.pth"
        prov_map_path_thai = self.weights_dir / "province_map.json"
        with open(prov_map_path_thai, "r", encoding="utf-8") as f:
            self.prov_map_thai = {int(k): v for k, v in json.load(f).items()}
        self.prov_model_thai = ResNetProvinceClassifier(n_classes=len(self.prov_map_thai), backbone="resnet18", pretrained=False)
        ckpt_p_th = torch.load(prov_path_thai, map_location=self.device)
        self.prov_model_thai.load_state_dict(ckpt_p_th.get("model_state", ckpt_p_th))
        self.prov_model_thai.to(self.device).eval()

        self.tf_prov_thai = get_grayscale_prov_transforms(is_train=False, size=(256, 80), use_ambient_padding=True, use_grayscale=True)
        self.tf_prov_lao = get_grayscale_prov_transforms(is_train=False, size=(256, 64), use_ambient_padding=True, use_grayscale=True)
        self.tf_char = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Thai CTC OCR Fallback Model (ResNetCRNN)
        from src.models import ResNetCRNN, best_path_decode
        from src.preprocess import get_ocr_transforms
        self.best_path_decode = best_path_decode
        self.ocr_model = None
        ocr_path = self.weights_dir / "ocr_model.pth"
        int_to_char_path = self.weights_dir / "int_to_char.json"
        if ocr_path.exists() and int_to_char_path.exists():
            with open(int_to_char_path, "r", encoding="utf-8") as f:
                self.int_to_char = {int(k): v for k, v in json.load(f).items()}
            self.ocr_model = ResNetCRNN(img_channel=1, num_classes=len(self.int_to_char))
            ckpt_ocr = torch.load(ocr_path, map_location=self.device)
            state = ckpt_ocr.get("model_state_dict", ckpt_ocr.get("model_state", ckpt_ocr))
            self.ocr_model.load_state_dict(state)
            self.ocr_model.to(self.device).eval()
            self.tf_ocr = get_ocr_transforms(is_train=False)
            print(f"  [8b/8] Thai CTC OCR (Fallback) : {ocr_path.name}")
        print("=" * 70 + "\n")

    def warp_plate(self, raw_bgr: np.ndarray, plate_box: List[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Executes homography perspective warp matching trained plate geometry.
        Returns (rectified_plate, quad_corners)."""
        bx1, by1, bx2, by2 = plate_box
        H, W = raw_bgr.shape[:2]
        bw = bx2 - bx1
        bh = by2 - by1

        if (self.corner_onnx_sess is not None or self.plate_corner_model is not None) and bw >= 15 and bh >= 15:
            try:
                # 2% subtle margin prevents edge clipping while strictly avoiding outer truck bumper/bracket capture
                pad_w = int(bw * 0.02)
                pad_h = int(bh * 0.02)
                c_x1 = max(0, bx1 - pad_w)
                c_y1 = max(0, by1 - pad_h)
                c_x2 = min(W, bx2 + pad_w)
                c_y2 = min(H, by2 + pad_h)
                crop_w = c_x2 - c_x1
                crop_h = c_y2 - c_y1

                crop_bgr = raw_bgr[c_y1:c_y2, c_x1:c_x2]
                if crop_bgr.size > 0:
                    crop_resized = cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    c_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    c_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    norm_c = ((crop_rgb - c_mean) / c_std).transpose(2, 0, 1)[np.newaxis, :]

                    if self.corner_onnx_sess is not None:
                        norm_corners = self.corner_onnx_sess.run(None, {self.corner_onnx_input: norm_c})[0].reshape(4, 2)
                    else:
                        with torch.no_grad():
                            ts_corner = torch.from_numpy(norm_c).to(self.device)
                            norm_corners = self.plate_corner_model(ts_corner).squeeze(0).cpu().numpy().reshape(4, 2)

                    quad_corners = np.array([
                        [c_x1 + cx * crop_w, c_y1 + cy * crop_h]
                        for cx, cy in norm_corners
                    ], dtype=np.float32)

                    # Homography warp with 6% safety margin to preserve plate borders and top banners
                    rectified_plate = warp_perspective_plate(
                        raw_bgr, quad_corners, target_width=320, target_height=160, padding_frac=0.06
                    )
                    if rectified_plate is not None and rectified_plate.size > 0:
                        return rectified_plate, quad_corners
            except Exception as e:
                print(f"[Model 1 Homography] Warning: {e}")

        # Fallback to direct resize
        raw_box = raw_bgr[by1:by2, bx1:bx2]
        fallback_quad = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.float32)
        return cv2.resize(raw_box, (320, 160), interpolation=cv2.INTER_CUBIC), fallback_quad

    def run_image(self, raw_bgr: np.ndarray, filename: str = "") -> Dict[str, Any]:
        """Runs the complete end-to-end recognition pipeline on a raw scene image."""
        t_start = time.time()
        H, W = raw_bgr.shape[:2]

        # -----------------------------------------------------------------
        # Stage 1: Model 1 Plate Detection
        # -----------------------------------------------------------------
        t1_start = time.time()
        if self.m1_onnx_sess is not None:
            mean_320 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std_320 = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            rgb_320 = cv2.resize(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), (320, 320)).astype(np.float32)
            blob_m1 = ((rgb_320 - mean_320) / std_320).transpose(2, 0, 1)[np.newaxis, :]
            out_m1 = self.m1_onnx_sess.run(None, {self.m1_onnx_input: blob_m1})[0]

            best_box = None
            best_conf = 0.0
            for row in out_m1[0]:
                score = float(row[4])
                if score > best_conf and score >= 0.25:
                    best_conf = score
                    x1 = max(0, min(W, int(round(row[0] * (W / 320.0)))))
                    y1 = max(0, min(H, int(round(row[1] * (H / 320.0)))))
                    x2 = max(0, min(W, int(round(row[2] * (W / 320.0)))))
                    y2 = max(0, min(H, int(round(row[3] * (H / 320.0)))))
                    best_box = [x1, y1, x2, y2]
            t_m1 = int((time.time() - t1_start) * 1000)

            if best_box is None:
                return {
                    "detected": False,
                    "message": "No license plate detected by Model 1",
                    "timings": {"m1_ms": t_m1, "total_ms": int((time.time() - t_start) * 1000)},
                    "raw_bgr": raw_bgr,
                }
            scene_plate_box = best_box
            plate_conf = best_conf
        else:
            res1 = self.m1_detector.predict(raw_bgr, conf=0.25, device=self.device)
            t_m1 = int((time.time() - t1_start) * 1000)

            if len(res1.boxes) == 0:
                return {
                    "detected": False,
                    "message": "No license plate detected by Model 1",
                    "timings": {"m1_ms": t_m1, "total_ms": int((time.time() - t_start) * 1000)},
                    "raw_bgr": raw_bgr,
                }

            best_box_idx = int(np.argmax([float(b.conf[0]) for b in res1.boxes]))
            scene_plate_box = res1.boxes[best_box_idx].xyxy[0].cpu().numpy().astype(int).tolist()
            plate_conf = float(res1.boxes[best_box_idx].conf[0])
            scene_x1, scene_y1, scene_x2, scene_y2 = scene_plate_box
            scene_x1, scene_y1 = max(0, scene_x1), max(0, scene_y1)
            scene_x2, scene_y2 = min(W, scene_x2), min(H, scene_y2)
            scene_plate_box = [scene_x1, scene_y1, scene_x2, scene_y2]

        # -----------------------------------------------------------------
        # Stage 2: Plate 4-Corner Homography Perspective Rectification
        # -----------------------------------------------------------------
        t_warp_start = time.time()
        rectified_plate, quad_corners = self.warp_plate(raw_bgr, scene_plate_box)
        t_warp = int((time.time() - t_warp_start) * 1000)

        if rectified_plate is None or rectified_plate.size == 0 or rectified_plate.shape[0] < 8:
            return {
                "detected": False,
                "message": "Failed to rectify plate crop",
                "timings": {"m1_ms": t_m1, "warp_ms": t_warp, "total_ms": int((time.time() - t_start) * 1000)},
                "raw_bgr": raw_bgr,
            }

        # -----------------------------------------------------------------
        # Stage 1.5: Country Classification (Thai vs Laos)
        # -----------------------------------------------------------------
        tc_start = time.time()
        if self.country_onnx_sess is not None:
            rgb_country = cv2.resize(cv2.cvtColor(rectified_plate, cv2.COLOR_BGR2RGB), (256, 128)).astype(np.float32) / 255.0
            c_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            c_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            blob_country = ((rgb_country - c_mean) / c_std).transpose(2, 0, 1)[np.newaxis, :]
            out_country = self.country_onnx_sess.run(None, {self.country_onnx_input: blob_country})[0][0]
            exp_c = np.exp(out_country - np.max(out_country))
            probs_c = exp_c / np.sum(exp_c)
            c_idx = int(np.argmax(probs_c))
            country_name = "Thai" if c_idx == 0 else "Laos"
            country_conf = float(probs_c[c_idx])
        else:
            pil_plate = Image.fromarray(cv2.cvtColor(rectified_plate, cv2.COLOR_BGR2RGB))
            ts_c = self.tf_country(pil_plate).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out_c = self.country_model(ts_c)
                prob_c = F.softmax(out_c, dim=1).squeeze(0)
                c_conf, c_idx = prob_c.max(0)
            country_name = "Thai" if c_idx.item() == 0 else "Laos"
            country_conf = float(c_conf.item())
        t_country = int((time.time() - tc_start) * 1000)

        rw, rh = rectified_plate.shape[1], rectified_plate.shape[0]
        char_details = []
        prov_box = None
        prov_name = "Unknown"
        prov_conf = 0.0
        prov_crop = None
        t_m2 = 0
        t_m3a = 0
        t_rec = 0
        t_prov = 0

        # =================================================================
        # Branch A: Thai License Plate Workflow
        # =================================================================
        if country_name == "Thai":
            t2_start = time.time()
            res2 = self.m2_detector.predict(rectified_plate, conf=0.15, device=self.device)
            t_m2 = int((time.time() - t2_start) * 1000)

            char_crop = None
            char_box_coords = None
            prov_crop = None
            prov_box = None
            best_char_conf = 0.0
            best_prov_conf = 0.0

            if len(res2.boxes) > 0:
                char_cands = []
                for b in res2.boxes:
                    c_idx_comp = int(b.cls[0])
                    bconf = float(b.conf[0])
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                    bx1 = max(0, min(rw, bx1))
                    bx2 = max(0, min(rw, bx2))
                    by1 = max(0, min(rh, by1))
                    by2 = max(0, min(rh, by2))

                    if c_idx_comp == 0 and by1 < int(rh * 0.65):
                        bw_char = bx2 - bx1
                        bh_char = by2 - by1
                        char_cands.append((bconf, bw_char, bh_char, bx1, by1, bx2, by2))
                    elif c_idx_comp == 1 and by2 > int(rh * 0.45) and ((by1 + by2) / 2) > int(rh * 0.40) and bconf > best_prov_conf:
                        pad_x = min(14, max(6, int((bx2 - bx1) * 0.08)))
                        pad_y = min(8, max(4, int((by2 - by1) * 0.10)))
                        pvx1 = max(0, bx1 - pad_x)
                        pvx2 = min(rw, bx2 + pad_x)
                        pvy1 = max(int(rh * 0.40), by1 - pad_y)
                        pvy2 = min(rh, by2 + pad_y)
                        prov_crop = rectified_plate[pvy1:pvy2, pvx1:pvx2]
                        prov_box = [pvx1, pvy1, pvx2, pvy2]
                        best_prov_conf = bconf

                # Discard top banners (e.g. THAILAND 73 on commercial trucks) when a genuine character row exists
                if char_cands:
                    valid_cands = [c for c in char_cands if c[2] >= 35 and c[1] >= 100]
                    chosen_pool = valid_cands if valid_cands else char_cands
                    chosen = max(chosen_pool, key=lambda c: c[0])
                    bconf, bw_char, bh_char, bx1, by1, bx2, by2 = chosen

                    pad_cx = max(10, int(bw_char * 0.05))
                    # Generous vertical margins: allow plenty of headroom and footroom
                    # to match the user annotations where character boxes extend beyond top/bottom glyph ink.
                    pad_cy_top = max(14, int(bh_char * 0.20))
                    pad_cy_bottom = max(18, int(bh_char * 0.22))
                    cx1 = max(0, bx1 - pad_cx)
                    cx2 = min(rw, bx2 + pad_cx)
                    cy1 = max(int(rh * 0.02), by1 - pad_cy_top)
                    # Keep below province area: Thai province starts at >= 0.72*rh
                    cy2 = min(int(rh * 0.74), by2 + pad_cy_bottom)
                    char_crop = rectified_plate[cy1:cy2, cx1:cx2]
                    char_box_coords = [cx1, cy1, cx2, cy2]
                    best_char_conf = bconf

            if char_crop is None:
                char_crop = rectified_plate[0:int(rh * 0.75), 0:rw]
                char_box_coords = [0, 0, rw, int(rh * 0.75)]
            if prov_crop is None:
                # Fallback: province is centered in the bottom band below character ink
                if char_box_coords is not None:
                    prov_y1 = max(int(rh * 0.72), char_box_coords[3] - 2)
                else:
                    prov_y1 = int(rh * 0.70)
                prov_box = [int(rw * 0.22), min(prov_y1, int(rh * 0.78)), int(rw * 0.78), int(rh * 0.98)]
                prov_crop = rectified_plate[prov_box[1]:prov_box[3], prov_box[0]:prov_box[2]]

            # Thai Character Box Detection
            t3a_start = time.time()
            if self.thai_charbox_detector is not None and char_crop.size > 0:
                ch_w, ch_h = char_crop.shape[1], char_crop.shape[0]
                # Contract: Thai PicoDet-S is trained with 30% vertical padding to prevent
                # Stride-16 DFL reg_max=7 saturation. Match serving geometry exactly:
                pad_y = int(round(ch_h * 0.30))
                padded_char_crop = cv2.copyMakeBorder(char_crop, pad_y, pad_y, 0, 0, cv2.BORDER_REPLICATE)

                # Use conf=0.50 to filter low-confidence jitter / slivers / inner loops
                res_tbox = self.thai_charbox_detector.predict(padded_char_crop, conf=0.50, device=self.device)
                raw_boxes = []
                for b in res_tbox.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                    bconf = float(b.conf[0])
                    # Unpad y-coordinates back to char_crop space
                    orig_by1 = max(0, min(ch_h, by1 - pad_y))
                    orig_by2 = max(0, min(ch_h, by2 - pad_y))
                    orig_bx1 = max(0, min(ch_w, bx1))
                    orig_bx2 = max(0, min(ch_w, bx2))

                    # Map to rectified plate coordinates
                    fx1 = max(0, min(rw, char_box_coords[0] + orig_bx1))
                    fy1 = max(0, min(rh, char_box_coords[1] + orig_by1))
                    fx2 = max(0, min(rw, char_box_coords[0] + orig_bx2))
                    fy2 = max(0, min(rh, char_box_coords[1] + orig_by2))
                    bw = fx2 - fx1
                    bh = fy2 - fy1
                    if bw >= 8 and bh >= 16:
                        raw_boxes.append([fx1, fy1, fx2, fy2, bconf])

                if raw_boxes:
                    # Sort by detection confidence descending
                    raw_boxes.sort(key=lambda b: b[4], reverse=True)

                    # Containment / Sub-stroke Suppression (IoS & Enclosure)
                    # Prevents small inner loops inside numbers (like 'บ' inside bottom loop of '3')
                    deduped_boxes = []
                    for b in raw_boxes:
                        bx1, by1, bx2, by2, bconf = b
                        bw = bx2 - bx1
                        bh = by2 - by1
                        b_area = bw * bh
                        is_substroke = False
                        for k in deduped_boxes:
                            kx1, ky1, kx2, ky2, kconf = k
                            kw = kx2 - kx1
                            kh = ky2 - ky1
                            k_area = kw * kh

                            ix1, iy1 = max(bx1, kx1), max(by1, ky1)
                            ix2, iy2 = min(bx2, kx2), min(by2, ky2)
                            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                            iou = inter / float(b_area + k_area - inter) if (b_area + k_area - inter) > 0 else 0
                            ios = inter / float(min(b_area, k_area)) if min(b_area, k_area) > 0 else 0
                            is_enclosed_x = (bx1 >= kx1 - 4 and bx2 <= kx2 + 4 and bw < 0.80 * kw)

                            if iou > 0.25 or ios > 0.60 or is_enclosed_x:
                                is_substroke = True
                                break
                        if not is_substroke:
                            deduped_boxes.append(b)

                    # Filter fragments and narrow slivers by median character width/height
                    if len(deduped_boxes) >= 2:
                        high_conf_w = [b[2] - b[0] for b in deduped_boxes if b[4] >= 0.70]
                        if high_conf_w:
                            med_w = np.median(high_conf_w)
                            deduped_boxes = [b for b in deduped_boxes if not ((b[2] - b[0]) < 0.65 * med_w and b[4] < 0.80)]
                        med_h = np.median([b[3] - b[1] for b in deduped_boxes])
                        char_boxes = [b for b in deduped_boxes if (b[3] - b[1]) >= 0.55 * med_h]
                    else:
                        char_boxes = deduped_boxes

                    char_boxes = sorted(char_boxes, key=lambda b: b[0])
                else:
                    char_boxes = []
            else:
                char_boxes = []
            t_m3a = int((time.time() - t3a_start) * 1000)

            # Thai Character Classification
            trec_start = time.time()
            n_thai_boxes = len(char_boxes)
            for idx_b, b in enumerate(char_boxes):
                bx1, by1, bx2, by2, bconf = b
                single = rectified_plate[by1:by2, bx1:bx2]
                if single.size == 0 or single.shape[0] < 4 or single.shape[1] < 4:
                    continue

                # Autocontrast enhancement before pad_to_square (mirrors train contract and api_server)
                pil_sc = Image.fromarray(cv2.cvtColor(single, cv2.COLOR_BGR2RGB))
                pil_sc = ImageOps.autocontrast(pil_sc, cutoff=2)
                single_enhanced = cv2.cvtColor(np.array(pil_sc), cv2.COLOR_RGB2BGR)

                # Ambient padding to square: preserves natural character aspect ratio (v2 audit contract)
                sh, sw = single_enhanced.shape[:2]
                smax = max(sh, sw)
                corners = np.array([single_enhanced[0, 0], single_enhanced[0, -1], single_enhanced[-1, 0], single_enhanced[-1, -1]])
                bg_col = np.median(corners, axis=0).astype(np.uint8)
                padded_c = np.full((smax, smax, 3), bg_col, dtype=np.uint8)
                padded_c[(smax - sh) // 2 : (smax - sh) // 2 + sh, (smax - sw) // 2 : (smax - sw) // 2 + sw] = single_enhanced

                pil_c = Image.fromarray(cv2.cvtColor(padded_c, cv2.COLOR_BGR2RGB))
                ts_c = self.tf_char(pil_c).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out_tc = self.thai_char_classifier(ts_c)
                    probs_tc = F.softmax(out_tc, dim=1).squeeze(0)
                    top_tc, top_ti = probs_tc.max(0)
                sym = self.char_map_thai.get(top_ti.item(), "?")
                char_details.append({
                    "char": sym,
                    "prob": round(float(top_tc.item()) * 100, 1),
                    "box": [bx1, by1, bx2, by2],
                    "box_conf": round(bconf, 3),
                })

            raw_chars_str = "".join([c["char"] for c in char_details])
            formatted_plate = format_thai_plate(raw_chars_str)
            is_valid = is_valid_thai_plate(formatted_plate)

            # Run CTC OCR on char_crop for fallback / fusion if box prediction is invalid or incomplete
            if self.ocr_model is not None and char_crop.size > 0:
                char_pil = Image.fromarray(cv2.cvtColor(char_crop, cv2.COLOR_BGR2RGB)).convert("L")
                ts_ocr = self.tf_ocr(char_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out_ocr = self.ocr_model(ts_ocr)
                    probs_ocr = out_ocr.softmax(-1)
                    ocr_text = self.best_path_decode(probs_ocr, self.int_to_char)[0]
                formatted_ocr = format_thai_plate(ocr_text)
                valid_ocr = is_valid_thai_plate(formatted_ocr)
                if not is_valid and valid_ocr:
                    formatted_plate = formatted_ocr
                    raw_chars_str = ocr_text
                    is_valid = True
                elif len(char_details) == 0 and formatted_ocr:
                    formatted_plate = formatted_ocr
                    raw_chars_str = ocr_text
                    is_valid = valid_ocr

            t_rec = int((time.time() - trec_start) * 1000)

            # Thai Province Classification
            tprov_start = time.time()
            if prov_crop is not None and prov_crop.size > 0 and prov_crop.shape[0] >= 4:
                pil_tp = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))
                ts_tp = self.tf_prov_thai(pil_tp).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out_tp = self.prov_model_thai(ts_tp)
                    probs_tp = F.softmax(out_tp, dim=1).squeeze(0)
                    top_tp, top_tpi = probs_tp.max(0)
                prov_name = self.prov_map_thai.get(top_tpi.item(), "Unknown")
                prov_conf = float(top_tp.item()) * 100
            t_prov = int((time.time() - tprov_start) * 1000)
            is_valid = is_valid_thai_plate(formatted_plate)

        # =================================================================
        # Branch B: Lao License Plate Workflow
        # =================================================================
        else:
            # 1. Exact Preprocessing & Padding Contract (src/scratch/build_clean_lao_datasets.py:321-323)
            # The Lao M3A character box detector and province detector were trained on 160h plates
            # padded with pad_y=80 (top/bottom) and pad_x=284 (left/right) -> 320 x (W1 + 568).
            pad_y = 80
            pad_x = 284
            plate_padded = cv2.copyMakeBorder(rectified_plate, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
            h_out, w_out = plate_padded.shape[:2]

            # 2. Lao Model 3A Character Box Detection
            t3a_start = time.time()
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
            rgb_320 = cv2.resize(cv2.cvtColor(plate_padded, cv2.COLOR_BGR2RGB), (320, 320)).astype(np.float32)
            blob_320 = ((rgb_320 - mean) / std).transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)

            m3a_out = self.m3a_sess.run(None, {self.m3a_input_name: blob_320})[0]

            raw_boxes = []
            for row in m3a_out[0]:
                x1, y1, x2, y2, score = row[:5]
                if score >= 0.35:
                    bx1_p = x1 * (w_out / 320.0)
                    by1_p = y1 * (h_out / 320.0)
                    bx2_p = x2 * (w_out / 320.0)
                    by2_p = y2 * (h_out / 320.0)

                    # Map coordinates back to 320x160 rectified plate
                    orig_x1 = max(0, min(rw, int(round(bx1_p - pad_x))))
                    orig_y1 = max(0, min(rh, int(round(by1_p - pad_y))))
                    orig_x2 = max(0, min(rw, int(round(bx2_p - pad_x))))
                    orig_y2 = max(0, min(rh, int(round(by2_p - pad_y))))
                    bw_char = orig_x2 - orig_x1
                    bh_char = orig_y2 - orig_y1

                    if score >= 0.40:
                        orig_x1 = max(0, min(rw, int(round(bx1_p - pad_x))))
                        orig_y1 = max(0, min(rh, int(round(by1_p - pad_y))))
                        orig_x2 = max(0, min(rw, int(round(bx2_p - pad_x))))
                        orig_y2 = max(0, min(rh, int(round(by2_p - pad_y))))
                        bw_char = orig_x2 - orig_x1
                        bh_char = orig_y2 - orig_y1

                        # True Lao characters on 320x160 plates have height >= 45px and start before y=105
                        if bw_char >= 12 and bh_char >= 45 and orig_y1 < 105:
                            sub_patch = rectified_plate[orig_y1:orig_y2, orig_x1:orig_x2]
                            if sub_patch.size > 0:
                                g_std = float(np.std(cv2.cvtColor(sub_patch, cv2.COLOR_BGR2GRAY)))
                                if g_std >= 2.0:
                                    raw_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2, float(score)])

            # Clean Standard IoU NMS (iou > 0.28 eliminates anchor duplicates while preserving close characters)
            if raw_boxes:
                raw_boxes.sort(key=lambda b: b[4], reverse=True)
                kept = []
                for b in raw_boxes:
                    ba = (b[2] - b[0]) * (b[3] - b[1])
                    overlap = False
                    for kb in kept:
                        ka = (kb[2] - kb[0]) * (kb[3] - kb[1])
                        iw = max(0, min(b[2], kb[2]) - max(b[0], kb[0]))
                        ih = max(0, min(b[3], kb[3]) - max(b[1], kb[1]))
                        inter = iw * ih
                        iou = inter / (ba + ka - inter) if (ba + ka - inter) > 0 else 0
                        if iou > 0.28:
                            overlap = True
                            break
                    if not overlap:
                        kept.append(b)
                char_boxes = sorted(kept, key=lambda b: b[0])
            else:
                char_boxes = []
            t_m3a = int((time.time() - t3a_start) * 1000)

            # 3. Lao Character Recognition (MobileNetV4 Lao v7 with Exact Preprocessing Contract)
            trec_start = time.time()
            n_lao_boxes = len(char_boxes)
            for idx_b, b in enumerate(char_boxes):
                bx1, by1, bx2, by2, bconf = b
                # Adaptive margin cushion with safe frame guard (do not cut into outer plate border)
                cx1 = max(0, bx1 - 2)
                cx2 = min(rw, bx2 + 2)
                cy1 = max(0, by1 - 1)
                cy2 = min(int(rh * 0.938), by2 + 2)
                c_crop = rectified_plate[cy1:cy2, cx1:cx2].copy()
                if c_crop.shape[0] < 4 or c_crop.shape[1] < 4:
                    continue

                sh, sw = c_crop.shape[:2]
                smax = max(sh, sw)
                corners = np.array([c_crop[0, 0], c_crop[0, -1], c_crop[-1, 0], c_crop[-1, -1]])
                bg_col = np.median(corners, axis=0).astype(np.uint8)
                padded_c = np.full((smax, smax, 3), bg_col, dtype=np.uint8)
                padded_c[(smax - sh) // 2 : (smax - sh) // 2 + sh, (smax - sw) // 2 : (smax - sw) // 2 + sw] = c_crop

                pil_sc = ImageOps.autocontrast(Image.fromarray(cv2.cvtColor(padded_c, cv2.COLOR_BGR2RGB)), cutoff=2)
                rgb_224 = cv2.resize(np.array(pil_sc), (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                blob_rec = ((rgb_224 - mean) / std).transpose(2, 0, 1)[np.newaxis, :]
                out_rec = self.rec_sess.run(None, {self.rec_input_name: blob_rec})[0][0].copy()

                # Spatial & Sequence Grammar Mask:
                # Lao plates: first 2 characters are consonants (indices >= 10 in v6 map).
                # Trailing 1-4 characters are strictly numeric digits (indices 0..9).
                cx = (bx1 + bx2) / 2.0
                is_trailing_digit = (idx_b >= 2) or (cx >= 0.38 * rw)
                if not is_trailing_digit and idx_b < 2:
                    out_rec[:10] = -1e9
                else:
                    out_rec[10:] = -1e9

                exp_v = np.exp(out_rec - np.max(out_rec))
                probs = exp_v / np.sum(exp_v)
                top_idx = int(np.argmax(probs))
                char_sym = self.char_map_lao.get(top_idx, "?")
                char_prob = float(probs[top_idx])

                char_details.append({
                    "char": char_sym,
                    "prob": round(char_prob * 100, 1),
                    "box": [bx1, by1, bx2, by2],
                    "box_conf": round(bconf, 3),
                })
            t_rec = int((time.time() - trec_start) * 1000)

            # 4. Lao Province Detection / Classification
            tprov_start = time.time()
            if self.prov_mode == "1stage":
                # 1-Stage Mode: PicoDet-S directly predicts the province class from the 320x160 plate
                if self.lao_prov_det_onnx_sess is not None:
                    rgb_320_prov = cv2.resize(cv2.cvtColor(rectified_plate, cv2.COLOR_BGR2RGB), (320, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    mean_p = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                    std_p = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                    blob_p = ((rgb_320_prov - mean_p) / std_p).transpose(2, 0, 1)[np.newaxis, :]
                    out_lp = self.lao_prov_det_onnx_sess.run(None, {self.lao_prov_det_onnx_input: blob_p})[0]
                    best_sc = -1.0
                    best_c = -1
                    best_bx = None
                    for row in out_lp[0]:
                        scores = row[4:]
                        c = int(np.argmax(scores))
                        sc = float(scores[c])
                        if sc > best_sc and sc >= 0.10:
                            best_sc = sc
                            best_c = c
                            x1 = max(0, min(rw, int(round(row[0] * (rw / 320.0)))))
                            y1 = max(0, min(rh, int(round(row[1] * (rh / 320.0)))))
                            x2 = max(0, min(rw, int(round(row[2] * (rw / 320.0)))))
                            y2 = max(0, min(rh, int(round(row[3] * (rh / 320.0)))))
                            best_bx = [x1, y1, x2, y2]
                    if best_c >= 0:
                        prov_box = best_bx
                        prov_name = self.prov_map_lao.get(best_c, f"Class_{best_c}")
                        prov_conf = best_sc * 100.0
                elif self.lao_prov_detector is not None:
                    res_lp = self.lao_prov_detector.predict(rectified_plate, conf=0.10, device=self.device)
                    if len(res_lp.boxes) > 0:
                        best_lp = max(res_lp.boxes, key=lambda b: float(b.conf[0]))
                        lpx1, lpy1, lpx2, lpy2 = best_lp.xyxy[0].tolist()
                        prov_box = [int(round(lpx1)), int(round(lpy1)), int(round(lpx2)), int(round(lpy2))]
                        cls_idx = int(best_lp.cls[0])
                        prov_name = self.prov_map_lao.get(cls_idx, f"Class_{cls_idx}")
                        prov_conf = float(best_lp.conf[0]) * 100
                if prov_box is not None:
                    prov_crop = rectified_plate[prov_box[1]:prov_box[3], prov_box[0]:prov_box[2]]
            else:
                # 2-Stage Mode:
                # Stage 1: Detect province banner bounding box (using plate directly)
                if self.lao_prov_det_onnx_sess is not None:
                    rgb_320_prov = cv2.resize(cv2.cvtColor(rectified_plate, cv2.COLOR_BGR2RGB), (320, 320), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    mean_p = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                    std_p = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                    blob_p = ((rgb_320_prov - mean_p) / std_p).transpose(2, 0, 1)[np.newaxis, :]
                    out_lp = self.lao_prov_det_onnx_sess.run(None, {self.lao_prov_det_onnx_input: blob_p})[0]
                    best_sc = -1.0
                    for row in out_lp[0]:
                        scores = row[4:]
                        sc = float(np.max(scores))
                        if sc > best_sc and sc >= 0.15:
                            best_sc = sc
                            pvx1 = max(0, min(rw, int(round(row[0] * (rw / 320.0)))))
                            pvy1 = max(0, min(rh, int(round(row[1] * (rh / 320.0)))))
                            pvx2 = max(0, min(rw, int(round(row[2] * (rw / 320.0)))))
                            pvy2 = max(0, min(rh, int(round(row[3] * (rh / 320.0)))))
                            if (pvx2 - pvx1) >= 20 and (pvy2 - pvy1) >= 8:
                                prov_box = [pvx1, pvy1, pvx2, pvy2]
                                prov_crop = rectified_plate[pvy1:pvy2, pvx1:pvx2]
                elif self.lao_prov_detector is not None:
                    res_lp = self.lao_prov_detector.predict(rectified_plate, conf=0.15, device=self.device)
                    if len(res_lp.boxes) > 0:
                        best_lp = max(res_lp.boxes, key=lambda b: float(b.conf[0]))
                        lpx1, lpy1, lpx2, lpy2 = best_lp.xyxy[0].tolist()
                        pvx1 = int(round(lpx1))
                        pvy1 = int(round(lpy1))
                        pvx2 = int(round(lpx2))
                        pvy2 = int(round(lpy2))
                        if (pvx2 - pvx1) >= 20 and (pvy2 - pvy1) >= 8:
                            prov_box = [pvx1, pvy1, pvx2, pvy2]
                            prov_crop = rectified_plate[pvy1:pvy2, pvx1:pvx2]

                # Fallback to upper banner geometry if detector missed
                if prov_box is None:
                    prov_box = [int(rw * 0.10), int(rh * 0.04), int(rw * 0.90), int(rh * 0.44)]
                    prov_crop = rectified_plate[prov_box[1]:prov_box[3], prov_box[0]:prov_box[2]]

                # Stage 2: High-Resolution Grayscale Province Classification (ResNet18 v6 at 256x64)
                if prov_crop is not None and prov_crop.size > 0 and prov_crop.shape[0] >= 4 and prov_crop.shape[1] >= 4:
                    if self.prov_model_lao_onnx_sess is not None:
                        gray = cv2.cvtColor(prov_crop, cv2.COLOR_BGR2GRAY)
                        gh, gw = gray.shape
                        corners_g = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
                        bg_val = int(np.median(corners_g))
                        ratio = min(256.0 / gw, 64.0 / gh)
                        nw = max(4, int(round(gw * ratio)))
                        nh = max(4, int(round(gh * ratio)))
                        resized_g = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
                        canvas = np.full((64, 256), bg_val, dtype=np.uint8)
                        px = (256 - nw) // 2
                        py = (64 - nh) // 2
                        canvas[py:py+nh, px:px+nw] = resized_g
                        norm_g = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.25
                        blob_g = np.stack([norm_g, norm_g, norm_g], axis=0)[np.newaxis, :]
                        out_p2 = self.prov_model_lao_onnx_sess.run(None, {self.prov_model_lao_onnx_input: blob_g})[0][0]
                        exp_p2 = np.exp(out_p2 - np.max(out_p2))
                        probs_p2 = exp_p2 / np.sum(exp_p2)
                        top_p2 = int(np.argmax(probs_p2))
                        prov_name = self.prov_map_lao.get(top_p2, "Unknown")
                        prov_conf = float(probs_p2[top_p2]) * 100
                    elif self.prov_model_lao is not None:
                        pil_lp = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))
                        ts_lp = self.tf_prov_lao(pil_lp).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            out_lp = self.prov_model_lao(ts_lp)
                            probs_lp = F.softmax(out_lp, dim=1).squeeze(0)
                            top_lp, top_lpi = probs_lp.max(0)
                        prov_name = self.prov_map_lao.get(top_lpi.item(), "Unknown")
                        prov_conf = float(top_lp.item()) * 100

            t_prov = int((time.time() - tprov_start) * 1000)
            raw_chars_str = "".join([c["char"] for c in char_details])
            formatted_plate = format_lao_plate(raw_chars_str)
            is_valid = is_valid_lao_plate(formatted_plate)

        t_total = int((time.time() - t_start) * 1000)

        return {
            "detected": True,
            "filename": filename,
            "plate_box": scene_plate_box,
            "plate_corners": quad_corners.tolist() if quad_corners is not None else None,
            "plate_conf": round(plate_conf, 3),
            "rectified_plate": rectified_plate,
            "country": country_name,
            "country_conf": round(country_conf, 4),
            "prov_box": prov_box,
            "prov_crop": prov_crop,
            "char_details": char_details,
            "raw_text": raw_chars_str,
            "plate_text": formatted_plate,
            "is_valid": is_valid,
            "province": prov_name,
            "province_conf": round(prov_conf, 1),
            "timings": {
                "m1_ms": t_m1,
                "warp_ms": t_warp,
                "country_ms": t_country,
                "m2_ms": t_m2,
                "m3a_ms": t_m3a,
                "rec_ms": t_rec,
                "prov_ms": t_prov,
                "total_ms": t_total,
            },
        }

    def render_dashboard(
        self,
        raw_bgr: np.ndarray,
        result: Dict[str, Any],
        idx: int,
        total_count: int,
    ) -> np.ndarray:
        """Composes a high-information dashboard visualization with overlays."""
        H, W = raw_bgr.shape[:2]
        canvas_h = 760
        canvas_w = 1200
        dashboard = np.full((canvas_h, canvas_w, 3), (24, 24, 28), dtype=np.uint8)

        # 1. Left View: Raw Scene with Plate Box (AABB + 4-Corner Quad)
        scene_w = 720
        scene_h = 540
        scene_disp = raw_bgr.copy()

        if result.get("detected"):
            px1, py1, px2, py2 = result["plate_box"]
            pconf = result["plate_conf"]

            # 1a. Axis-Aligned Bounding Box (AABB) in Green
            cv2.rectangle(scene_disp, (px1, py1), (px2, py2), (0, 220, 0), 2)

            # 1b. 4-Corner Homography Polygon (Quad) in Cyan with Corner Keypoints
            quad = result.get("plate_corners")
            if quad is not None:
                pts = np.array(quad, dtype=np.int32)
                cv2.polylines(scene_disp, [pts], isClosed=True, color=(0, 240, 255), thickness=2)
                for pt in pts:
                    cv2.circle(scene_disp, tuple(pt), 4, (0, 0, 255), -1)
                    cv2.circle(scene_disp, tuple(pt), 2, (255, 255, 255), -1)

            c_label = f"{result['country']} Plate {pconf:.2f} (Green: AABB | Cyan: 4-Corner)"
            scene_disp = draw_text_pil(
                scene_disp, c_label, (px1, max(0, py1 - 24)), font_size=15,
                text_color=(0, 0, 0), bg_color=(0, 220, 0)
            )

        # Scale scene image to fit left viewport
        scale_s = min(scene_w / float(W), scene_h / float(H))
        dw, dh = int(round(W * scale_s)), int(round(H * scale_s))
        resized_scene = cv2.resize(scene_disp, (dw, dh), interpolation=cv2.INTER_AREA)
        off_x = 20 + (scene_w - dw) // 2
        off_y = 60 + (scene_h - dh) // 2
        dashboard[off_y:off_y + dh, off_x:off_x + dw] = resized_scene
        cv2.rectangle(dashboard, (off_x - 1, off_y - 1), (off_x + dw + 1, off_y + dh + 1), (60, 60, 70), 1)

        # 2. Right Panels: Crops and Recognition details
        right_x = 760

        # Header Title
        title = f"Image [{idx + 1}/{total_count}]: {Path(result.get('filename', 'Unknown')).name}"
        dashboard = draw_text_pil(
            dashboard, title, (20, 16), font_size=20,
            text_color=(255, 255, 255)
        )

        if not result.get("detected"):
            dashboard = draw_text_pil(
                dashboard, "❌ NO PLATE DETECTED", (right_x, 100), font_size=24,
                text_color=(0, 0, 255)
            )
            return dashboard

        # 2a. Rectified Plate Crop Display with Character Boxes (Green) & Province Box (Yellow)
        rectified_plate = result["rectified_plate"]
        plate_disp_src = rectified_plate.copy()

        # Draw Province Box in Yellow
        prov_box = result.get("prov_box")
        if prov_box is not None:
            pvx1, pvy1, pvx2, pvy2 = prov_box
            cv2.rectangle(plate_disp_src, (pvx1, pvy1), (pvx2, pvy2), (0, 215, 255), 2)
            prov_txt = result.get("province", "")
            if prov_txt and prov_txt != "Unknown":
                plate_disp_src = draw_text_pil(
                    plate_disp_src, prov_txt, (pvx1, max(0, pvy1 - 16)), font_size=12,
                    text_color=(0, 0, 0), bg_color=(0, 215, 255), padding=1
                )

        # Draw Character Boxes in Green
        for item in result["char_details"]:
            cbx1, cby1, cbx2, cby2 = item["box"]
            cv2.rectangle(plate_disp_src, (cbx1, cby1), (cbx2, cby2), (0, 255, 0), 2)
            plate_disp_src = draw_text_pil(
                plate_disp_src, item["char"], (cbx1, max(0, cby1 - 16)), font_size=12,
                text_color=(0, 0, 0), bg_color=(0, 255, 0), padding=1
            )

        p_h, p_w = plate_disp_src.shape[:2]
        target_pw = 400
        target_ph = max(50, int(round(p_h * (target_pw / float(p_w)))))
        plate_disp = cv2.resize(plate_disp_src, (target_pw, min(160, target_ph)), interpolation=cv2.INTER_LINEAR)
        py_start = 80
        dashboard[py_start:py_start + plate_disp.shape[0], right_x:right_x + target_pw] = plate_disp
        cv2.rectangle(dashboard, (right_x - 1, py_start - 1), (right_x + target_pw + 1, py_start + plate_disp.shape[0] + 1), (80, 80, 100), 1)

        # Tag above plate crop
        tag_p = f"Warped Plate ({p_w}x{p_h}) | Yellow: Prov | Green: {len(result['char_details'])} Chars"
        dashboard = draw_text_pil(dashboard, tag_p, (right_x, py_start - 24), font_size=14, text_color=(200, 200, 200))

        # 2b. Province Crop Display
        prov_crop = result.get("prov_crop")
        if prov_crop is not None and prov_crop.size > 0:
            pr_h, pr_w = prov_crop.shape[:2]
            pr_disp_w = 400
            pr_disp_h = min(60, max(24, int(round(pr_h * (pr_disp_w / float(pr_w))))))
            prov_vis = cv2.resize(prov_crop, (pr_disp_w, pr_disp_h), interpolation=cv2.INTER_LINEAR)
            cy_start = py_start + plate_disp.shape[0] + 35
            dashboard[cy_start:cy_start + pr_disp_h, right_x:right_x + pr_disp_w] = prov_vis
            cv2.rectangle(dashboard, (right_x - 1, cy_start - 1), (right_x + pr_disp_w + 1, cy_start + pr_disp_h + 1), (80, 80, 100), 1)
            tag_c = f"Province Banner Crop ({pr_w}x{pr_h})"
            dashboard = draw_text_pil(dashboard, tag_c, (right_x, cy_start - 20), font_size=14, text_color=(200, 200, 200))
            sc_h = pr_disp_h
        else:
            cy_start = py_start + plate_disp.shape[0] + 10
            sc_h = 10

        # 2c. Final Recognition Summary Card
        card_y = cy_start + sc_h + 35
        card_w = 400
        card_h = 190
        cv2.rectangle(dashboard, (right_x, card_y), (right_x + card_w, card_y + card_h), (35, 36, 42), -1)
        cv2.rectangle(dashboard, (right_x, card_y), (right_x + card_w, card_y + card_h), (80, 80, 95), 1)

        # Country Banner
        c_tag = "[LAOS]" if result["country"] == "Laos" else "[THAI]"
        c_line = f"Country: {result['country']} {c_tag} ({result['country_conf'] * 100:.1f}%)"
        dashboard = draw_text_pil(dashboard, c_line, (right_x + 14, card_y + 14), font_size=16, text_color=(100, 220, 255))

        # Recognized Plate Text
        plate_text_disp = result["plate_text"] if result["plate_text"] else "(None)"
        valid_mark = "[VALID]" if result["is_valid"] else "[FORMAT CHECK]"
        p_line = f"Plate: {plate_text_disp}  {valid_mark}"
        dashboard = draw_text_pil(dashboard, p_line, (right_x + 14, card_y + 44), font_size=26, text_color=(0, 255, 120))

        # Province
        prov_disp = f"Province: {result['province']} ({result['province_conf']:.1f}%)"
        dashboard = draw_text_pil(dashboard, prov_disp, (right_x + 14, card_y + 92), font_size=18, text_color=(255, 215, 0))

        # Character Confidences List
        conf_strs = [f"{c['char']}:{c['prob']:.0f}%" for c in result["char_details"]]
        char_conf_line = "Chars: " + " ".join(conf_strs) if conf_strs else "Chars: (None)"
        dashboard = draw_text_pil(dashboard, char_conf_line, (right_x + 14, card_y + 128), font_size=14, text_color=(200, 200, 200))

        # Latency breakdown
        timings = result["timings"]
        lat_line = f"Latency: {timings['total_ms']}ms [M1:{timings['m1_ms']} Warp:{timings.get('warp_ms',0)} C:{timings['country_ms']} M3A:{timings['m3a_ms']} Rec:{timings['rec_ms']} Prov:{timings['prov_ms']}]"
        dashboard = draw_text_pil(dashboard, lat_line, (right_x + 14, card_y + 158), font_size=11, text_color=(150, 150, 160))

        # 3. Bottom Navigation Bar
        nav_y = canvas_h - 45
        cv2.rectangle(dashboard, (0, nav_y), (canvas_w, canvas_h), (18, 18, 22), -1)
        nav_text = "[SPACE] Next Image    [B] Previous Image    [S] Save Snapshot    [Q/ESC] Quit"
        dashboard = draw_text_pil(dashboard, nav_text, (20, nav_y + 10), font_size=15, text_color=(220, 220, 100))

        return dashboard


# -------------------------------------------------------------------------
# Interactive CLI Driver
# -------------------------------------------------------------------------
def collect_image_paths(path_str: str) -> List[str]:
    """Resolves a folder path or single image file into a list of valid image paths."""
    p = Path(path_str.strip().strip("'").strip('"'))
    if not p.exists():
        print(f"Error: Path does not exist: {p}")
        return []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if p.is_file():
        if p.suffix.lower() in exts:
            return [str(p.resolve())]
        else:
            print(f"Error: File is not a supported image: {p}")
            return []

    if p.is_dir():
        collected = []
        for file in sorted(p.iterdir()):
            if file.is_file() and file.suffix.lower() in exts:
                collected.append(str(file.resolve()))
        return collected

    return []


def print_cli_summary(res: Dict[str, Any], idx: int, total: int):
    """Prints a structured summary table for the current image to the terminal."""
    fname = Path(res.get("filename", "")).name
    t = res.get("timings", {})
    print("\n" + "─" * 70)
    print(f"[{idx + 1}/{total}] {fname}")
    print("─" * 70)
    if not res.get("detected"):
        print("  ❌ Plate: NOT DETECTED")
        print(f"  Latency: {t.get('total_ms', 0)} ms")
        return

    print(f"  Plate Box  : {res['plate_box']}  (Conf: {res['plate_conf']:.2f})")
    print(f"  Country    : {res['country']}  (Conf: {res['country_conf']*100:.1f}%)")
    print(f"  Plate Text : {res['plate_text']}  ({'Valid' if res['is_valid'] else 'Invalid Format'})")
    print(f"  Province   : {res['province']}  (Conf: {res['province_conf']:.1f}%)")
    print(f"  Timing     : Total {t.get('total_ms', 0)}ms [M1:{t.get('m1_ms', 0)}ms, Warp:{t.get('warp_ms', 0)}ms, Country:{t.get('country_ms', 0)}ms, M3A:{t.get('m3a_ms', 0)}ms, Rec:{t.get('rec_ms', 0)}ms, Prov:{t.get('prov_ms', 0)}ms]")
    print("  Characters :")
    for ci, c in enumerate(res.get("char_details", [])):
        print(f"    [{ci}] '{c['char']}' -> Conf: {c['prob']:5.1f}% | Box: {c['box']} (Det: {c['box_conf']:.2f})")
    print("─" * 70)


def main():
    parser = argparse.ArgumentParser(description="Lao & Thai LPR Full Pipeline Interactive Test Tool")
    parser.add_argument("input_path", nargs="?", type=str, default=None, help="Folder path or image path to test")
    parser.add_argument("--save-all", action="store_true", help="Save snapshots for all images non-interactively without GUI window")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory for saving snapshots")
    parser.add_argument("--prov-mode", type=str, choices=["1stage", "2stage"], default="2stage", help="Province prediction mode: '1stage' (PicoDet-S direct) or '2stage' (PicoDet-S box + ResNet18 classifier)")
    parser.add_argument("--onnx", dest="use_onnx", action="store_true", default=True, help="Run with 100% pure ONNX models (default: True)")
    parser.add_argument("--pytorch", "--no-onnx", dest="use_onnx", action="store_false", help="Run with PyTorch / LibreYOLO .pt checkpoints")
    args = parser.parse_args()

    input_path = args.input_path
    if not input_path:
        default_folder = str(cfg.PROJECT_ROOT / "datasets" / "Lao" / "Lao License Plates.v2i.yolov11" / "test" / "images")
        print("No path provided via CLI.")
        prompt_msg = f"Enter test folder or image path (Press ENTER for default: {default_folder}):\n> "
        user_in = input(prompt_msg).strip()
        input_path = user_in if user_in else default_folder

    image_paths = collect_image_paths(input_path)
    if not image_paths:
        print(f"No valid images found at: {input_path}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images to test.")

    tester = LaoFullPipelineTester(device="cpu", prov_mode=args.prov_mode, use_onnx=args.use_onnx)

    output_dir = Path(args.out_dir) if args.out_dir else (cfg.PROJECT_ROOT / "scratch" / "lao_test_snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    cur_idx = 0
    total = len(image_paths)

    if args.save_all:
        print(f"Running non-interactive batch test on {total} images -> saving to {output_dir}")
        for idx, img_path in enumerate(image_paths):
            raw_bgr = cv2.imread(img_path)
            if raw_bgr is None:
                continue
            result = tester.run_image(raw_bgr, filename=img_path)
            print_cli_summary(result, idx, total)
            dashboard = tester.render_dashboard(raw_bgr, result, idx, total)
            snap_path = output_dir / f"snapshot_{Path(img_path).stem}.png"
            cv2.imwrite(str(snap_path), dashboard)
            print(f"  📸 Saved snapshot [{idx + 1}/{total}]: {snap_path.name}")
        print("\nAll batch snapshots saved successfully!")
        return

    win_name = "LPR Full Pipeline Interactive Test Tool (Thai & Laos)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 760)

    while 0 <= cur_idx < total:
        img_path = image_paths[cur_idx]
        raw_bgr = cv2.imread(img_path)
        if raw_bgr is None:
            print(f"Warning: Failed to read image: {img_path}")
            cur_idx += 1
            continue

        result = tester.run_image(raw_bgr, filename=img_path)
        print_cli_summary(result, cur_idx, total)

        dashboard = tester.render_dashboard(raw_bgr, result, cur_idx, total)
        cv2.imshow(win_name, dashboard)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            print("\nExiting interactive test tool.")
            break
        elif key in (ord("b"), ord("B")):     # Previous image
            cur_idx = max(0, cur_idx - 1)
        elif key in (ord("s"), ord("S")):     # Save snapshot
            snap_path = output_dir / f"snapshot_{Path(img_path).stem}.png"
            cv2.imwrite(str(snap_path), dashboard)
            print(f"  📸 Saved snapshot to: {snap_path}")
        else:                                 # SPACE or any other key advances
            cur_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
