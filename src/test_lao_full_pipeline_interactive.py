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
        "/System/Library/Fonts/Supplemental/Lao Sangam MN.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Thonburi.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
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
    def __init__(self, device: str = "cpu", prov_mode: str = "2stage"):
        self.device = device
        self.prov_mode = prov_mode
        self.weights_dir = cfg.WEIGHTS_DIR
        print("\n" + "=" * 70)
        print(" Loading End-to-End Recognition Models (Thai & Lao) ".center(70, "="))
        print("=" * 70)

        # 1. Model 1: Plate Detector (PicoDet-S v6aug)
        m1_path = self.weights_dir / "plate_detector_picodet_s_v6aug.pt"
        if not m1_path.exists():
            m1_path = self.weights_dir / "plate_detector_picodet_s.pt"
        print(f"  [1/8] Model 1 (Plate Detector)  : {m1_path.name}")
        self.m1_detector = LibreYOLO(str(m1_path))

        # 2. Plate 4-Corner Homography Regressor (MobileNetV3-Small)
        corner_path = self.weights_dir / "plate_corner_regressor.pth"
        self.plate_corner_model = None
        if corner_path.exists():
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
        c_path = self.weights_dir / "country_classifier.pth"
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

        # 4. Model 2: Component Detector (PicoDet-S frame)
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
        prov_det_path = self.weights_dir / "province_detector_picodet_s_lao_v3.pt"
        if not prov_det_path.exists():
            prov_det_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "province_detector_picodet_s_lao_v3.pt"
        if not prov_det_path.exists():
            prov_det_path = self.weights_dir / "province_detector_picodet_s_lao_v2.pt"
        if not prov_det_path.exists():
            prov_det_path = cfg.PROJECT_ROOT / "chacrit" / "output model" / "province_detector_picodet_s_lao_v2.pt"
        self.lao_prov_detector = None
        if prov_det_path.exists():
            print(f"  [7/8] Lao Province Detector     : {prov_det_path.name}")
            try:
                self.lao_prov_detector = LibreYOLO(str(prov_det_path))
            except Exception as e:
                print(f"        Warning: Could not load Lao Province Detector: {e}")

        prov_path_lao = self.weights_dir / "province_model_grayscale_lao_v6.pth"
        if not prov_path_lao.exists():
            prov_path_lao = self.weights_dir / "province_model_grayscale_lao_v3.pth"
        prov_map_path_lao = self.weights_dir / "province_map_lao.json"
        with open(prov_map_path_lao, "r", encoding="utf-8") as f:
            self.prov_map_lao = {int(k): v for k, v in json.load(f).items()}
        print(f"  [7/8b] Lao 2-Stage Prov Clf    : {prov_path_lao.name} ({len(self.prov_map_lao)} classes)")
        self.prov_model_lao = ResNetProvinceClassifier(n_classes=len(self.prov_map_lao), backbone="resnet18", pretrained=False)
        ckpt_p_lao = torch.load(prov_path_lao, map_location=self.device)
        self.prov_model_lao.load_state_dict(ckpt_p_lao.get("model_state", ckpt_p_lao))
        self.prov_model_lao.to(self.device).eval()

        # 8. Thai Models (Fallback for Thai plates)
        thai_charbox_path = self.weights_dir / "character_box_detector_picodet_s_v6frame.pt"
        if not thai_charbox_path.exists():
            thai_charbox_path = self.weights_dir / "character_box_detector_picodet_s_v2.pt"
        print(f"  [8/8] Thai Models (M3A/Prov)    : {thai_charbox_path.name}")
        self.thai_charbox_detector = LibreYOLO(str(thai_charbox_path)) if thai_charbox_path.exists() else None

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
        print("=" * 70 + "\n")

    def warp_plate(self, raw_bgr: np.ndarray, plate_box: List[int]) -> np.ndarray:
        """Executes homography perspective warp matching trained plate geometry."""
        bx1, by1, bx2, by2 = plate_box
        H, W = raw_bgr.shape[:2]
        bw = bx2 - bx1
        bh = by2 - by1

        if self.plate_corner_model is not None and bw >= 15 and bh >= 15:
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
                    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                    ts_corner = self.tf_corner(crop_rgb).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        norm_corners = self.plate_corner_model(ts_corner).squeeze(0).cpu().numpy().reshape(4, 2)

                    quad_corners = np.array([
                        [c_x1 + cx * crop_w, c_y1 + cy * crop_h]
                        for cx, cy in norm_corners
                    ], dtype=np.float32)

                    # Exact homography warp without fake extra bumper expansion (padding_frac=0.0)
                    rectified_plate = warp_perspective_plate(
                        raw_bgr, quad_corners, target_width=320, target_height=160, padding_frac=0.0
                    )
                    if rectified_plate is not None and rectified_plate.size > 0:
                        return rectified_plate
            except Exception as e:
                print(f"[Model 1 Homography] Warning: {e}")

        # Fallback to direct resize
        raw_box = raw_bgr[by1:by2, bx1:bx2]
        return cv2.resize(raw_box, (320, 160), interpolation=cv2.INTER_CUBIC)

    def run_image(self, raw_bgr: np.ndarray, filename: str = "") -> Dict[str, Any]:
        """Runs the complete end-to-end recognition pipeline on a raw scene image."""
        t_start = time.time()
        H, W = raw_bgr.shape[:2]

        # -----------------------------------------------------------------
        # Stage 1: Model 1 Plate Detection
        # -----------------------------------------------------------------
        t1_start = time.time()
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
        plate_box = res1.boxes[best_box_idx].xyxy[0].cpu().numpy().astype(int)
        plate_conf = float(res1.boxes[best_box_idx].conf[0])
        px1, py1, px2, py2 = plate_box
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(W, px2), min(H, py2)

        # -----------------------------------------------------------------
        # Stage 2: Plate 4-Corner Homography Perspective Rectification
        # -----------------------------------------------------------------
        t_warp_start = time.time()
        rectified_plate = self.warp_plate(raw_bgr, [px1, py1, px2, py2])
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
            m2_input = cv2.resize(rectified_plate, (1280, 640))
            res2 = self.m2_detector.predict(m2_input, conf=0.20, device=self.device)
            t_m2 = int((time.time() - t2_start) * 1000)

            char_crop = None
            char_box_coords = None
            if len(res2.boxes) > 0:
                for b in res2.boxes:
                    c_idx_comp = int(b.cls[0])
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                    bx1 = max(0, min(rw, int(bx1 * rw / 1280.0)))
                    bx2 = max(0, min(rw, int(bx2 * rw / 1280.0)))
                    by1 = max(0, min(rh, int(by1 * rh / 640.0)))
                    by2 = max(0, min(rh, int(by2 * rh / 640.0)))

                    if c_idx_comp == 0 and by1 < int(rh * 0.65):  # Characters
                        pad_cx = min(10, max(3, int((bx2 - bx1) * 0.04)))
                        pad_cy = min(10, max(3, int((by2 - by1) * 0.10)))
                        cx1 = max(int(rw * 0.03), bx1 - pad_cx)
                        cx2 = min(int(rw * 0.97), bx2 + pad_cx)
                        cy1 = max(int(rh * 0.02), by1 - pad_cy)
                        cy2 = min(int(rh * 0.76), by2 + pad_cy)
                        char_crop = rectified_plate[cy1:cy2, cx1:cx2]
                        char_box_coords = [cx1, cy1, cx2, cy2]
                    elif c_idx_comp == 1 and by2 > int(rh * 0.50):  # Province
                        pad_x = min(14, max(6, int((bx2 - bx1) * 0.08)))
                        pad_y = min(8, max(4, int((by2 - by1) * 0.10)))
                        px1 = max(0, bx1 - pad_x)
                        px2 = min(rw, bx2 + pad_x)
                        py1 = max(int(rh * 0.46), by1 - pad_y)
                        py2 = min(rh, by2 + pad_y)
                        prov_crop = rectified_plate[py1:py2, px1:px2]
                        prov_box = [px1, py1, px2, py2]

            if char_crop is None:
                char_crop = rectified_plate[0:int(rh * 0.75), 0:rw]
                char_box_coords = [0, 0, rw, int(rh * 0.75)]
            if prov_crop is None:
                prov_box = [int(rw * 0.12), int(rh * 0.60), int(rw * 0.88), int(rh * 0.98)]
                prov_crop = rectified_plate[prov_box[1]:prov_box[3], prov_box[0]:prov_box[2]]

            # Thai Character Box Detection
            t3a_start = time.time()
            if self.thai_charbox_detector is not None and char_crop.size > 0:
                res_tbox = self.thai_charbox_detector.predict(char_crop, conf=0.15, device=self.device)
                ch_w, ch_h = char_crop.shape[1], char_crop.shape[0]
                raw_boxes = []
                for b in res_tbox.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                    bconf = float(b.conf[0])
                    # Map to rectified plate coordinates
                    fx1 = max(0, min(rw, char_box_coords[0] + bx1))
                    fy1 = max(0, min(rh, char_box_coords[1] + by1))
                    fx2 = max(0, min(rw, char_box_coords[0] + bx2))
                    fy2 = max(0, min(rh, char_box_coords[1] + by2))
                    if (fx2 - fx1) >= 6 and (fy2 - fy1) >= 8:
                        raw_boxes.append([fx1, fy1, fx2, fy2, bconf])

                if raw_boxes:
                    b_t = torch.tensor([[b[0], b[1], b[2], b[3]] for b in raw_boxes], dtype=torch.float32)
                    s_t = torch.tensor([b[4] for b in raw_boxes], dtype=torch.float32)
                    keep = nms(b_t, s_t, iou_threshold=0.35)
                    char_boxes = [raw_boxes[i] for i in keep.tolist()]
                    char_boxes = sorted(char_boxes, key=lambda b: b[0])
                else:
                    char_boxes = []
            else:
                char_boxes = []
            t_m3a = int((time.time() - t3a_start) * 1000)

            # Thai Character Classification
            trec_start = time.time()
            for b in char_boxes:
                bx1, by1, bx2, by2, bconf = b
                single = rectified_plate[by1:by2, bx1:bx2]
                if single.size == 0 or single.shape[0] < 4 or single.shape[1] < 4:
                    continue
                pil_c = Image.fromarray(cv2.cvtColor(single, cv2.COLOR_BGR2RGB))
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

            raw_chars_str = "".join([c["char"] for c in char_details])
            formatted_plate = format_thai_plate(raw_chars_str)
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
            for b in char_boxes:
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

                # Spatial Grammar Mask:
                # First ~38% plate width MUST be Lao consonants (indices >= 10 in v6 map).
                # Remaining ~62% plate width MUST be digits (indices 0..9).
                cx = (bx1 + bx2) / 2.0
                if cx < 0.38 * rw:
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
                # 1-Stage Mode: PicoDet-S directly predicts the province class from the plate
                if self.lao_prov_detector is not None:
                    res_lp = self.lao_prov_detector.predict(plate_padded, conf=0.10, device=self.device)
                    if len(res_lp.boxes) > 0:
                        best_lp = max(res_lp.boxes, key=lambda b: float(b.conf[0]))
                        lpx1, lpy1, lpx2, lpy2 = best_lp.xyxy[0].tolist()
                        orig_px1 = max(0, min(rw, int(round(lpx1 - pad_x))))
                        orig_px2 = max(0, min(rw, int(round(lpx2 - pad_x))))
                        orig_py1 = max(0, min(rh, int(round(lpy1 - pad_y))))
                        orig_py2 = max(0, min(rh, int(round(lpy2 - pad_y))))
                        prov_box = [orig_px1, orig_py1, orig_px2, orig_py2]
                        cls_idx = int(best_lp.cls[0])
                        prov_name = self.prov_map_lao.get(cls_idx, f"Class_{cls_idx}")
                        prov_conf = float(best_lp.conf[0]) * 100
                if prov_box is not None:
                    prov_crop = rectified_plate[prov_box[1]:prov_box[3], prov_box[0]:prov_box[2]]
            else:
                # 2-Stage Mode:
                # Stage 1: Detect province banner bounding box
                if self.lao_prov_detector is not None:
                    res_lp = self.lao_prov_detector.predict(plate_padded, conf=0.15, device=self.device)
                    if len(res_lp.boxes) > 0:
                        best_lp = max(res_lp.boxes, key=lambda b: float(b.conf[0]))
                        lpx1, lpy1, lpx2, lpy2 = best_lp.xyxy[0].tolist()
                        orig_px1 = max(0, min(rw, int(round(lpx1 - pad_x))))
                        orig_px2 = max(0, min(rw, int(round(lpx2 - pad_x))))
                        orig_py1 = max(0, min(rh, int(round(lpy1 - pad_y))))
                        orig_py2 = max(0, min(rh, int(round(lpy2 - pad_y))))
                        if (orig_px2 - orig_px1) >= 20 and (orig_py2 - orig_py1) >= 8:
                            prov_box = [orig_px1, orig_py1, orig_px2, orig_py2]
                            prov_crop = rectified_plate[orig_py1:orig_py2, orig_px1:orig_px2]

                # Fallback to upper banner geometry if detector missed
                if prov_box is None:
                    prov_box = [int(rw * 0.10), int(rh * 0.04), int(rw * 0.90), int(rh * 0.44)]
                    prov_crop = rectified_plate[prov_box[1]:prov_box[3], prov_box[0]:prov_box[2]]

                # Stage 2: High-Resolution Grayscale Province Classification (ResNet18 v6 at 256x64)
                if prov_crop is not None and prov_crop.size > 0 and prov_crop.shape[0] >= 4 and prov_crop.shape[1] >= 4:
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
            "plate_box": [px1, py1, px2, py2],
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

        # 1. Left View: Raw Scene with Plate Box
        scene_w = 720
        scene_h = 540
        scene_disp = raw_bgr.copy()

        if result.get("detected"):
            px1, py1, px2, py2 = result["plate_box"]
            pconf = result["plate_conf"]
            cv2.rectangle(scene_disp, (px1, py1), (px2, py2), (0, 220, 0), 2)
            c_label = f"{result['country']} Plate {pconf:.2f}"
            scene_disp = draw_text_pil(
                scene_disp, c_label, (px1, max(0, py1 - 24)), font_size=16,
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

    tester = LaoFullPipelineTester(device="cpu", prov_mode=args.prov_mode)

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
