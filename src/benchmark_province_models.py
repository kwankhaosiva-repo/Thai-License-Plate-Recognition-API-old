"""
src/benchmark_province_models.py

Benchmarks Province Recognition Approaches on branch obj-province:
  Approach 1: 2-Stage Component Crop (Model 2 / Banner) + Old RGB Classifier
  Approach 2: 2-Stage Component Crop + New Grayscale Classifier (ResNet18 + GrayscaleSmartResize)
  Approach 3: 1-Stage Direct Plate Object Detection (YOLO11)

Compares:
  - Classification Accuracy (Top-1 Match vs Ground Truth)
  - Inference Latency (ms/image)
  - Sensitivity to Model 2 bounding box cropping / missing component
"""

import sys
import time
import json
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from src.preprocess import get_prov_transforms
from src.models import ResNetProvinceClassifier


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


def get_grayscale_val_transforms():
    return transforms.Compose([
        GrayscaleSmartResize((256, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])


def benchmark_thai(num_test_samples=50):
    print("=" * 80)
    print("🇹🇭 Benchmarking Thai Province Recognition (3 Approaches Side-by-Side)")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m2_path = PROJECT_ROOT / "weights" / "component_detector.pt"
    old_cls_path = PROJECT_ROOT / "weights" / "province_model.pth"
    gray_cls_path = PROJECT_ROOT / "weights" / "province_model_grayscale_thai.pth"
    prov_map_path = PROJECT_ROOT / "weights" / "province_map.json"
    abbr_map_path = PROJECT_ROOT / "weights" / "province_abbr_map.json"
    yolo_model_path = PROJECT_ROOT / "weights" / "province_detector_yolo_thai.pt"

    test_img_dir = PROJECT_ROOT / "datasets" / "Thai" / "thai_province_yolo_balanced" / "test" / "images"
    test_lbl_dir = PROJECT_ROOT / "datasets" / "Thai" / "thai_province_yolo_balanced" / "test" / "labels"

    if not test_img_dir.exists():
        print(f"Test directory not found: {test_img_dir}")
        return

    with open(prov_map_path, "r", encoding="utf-8") as f:
        prov_map = json.load(f)
    with open(abbr_map_path, "r", encoding="utf-8") as f:
        abbr_map = json.load(f)

    abbr_to_name = {k: v["thai_name"] for k, v in abbr_map.items()}

    has_old_cls = old_cls_path.exists()
    has_gray_cls = gray_cls_path.exists()
    has_yolo = yolo_model_path.exists()

    print(f"1. Old RGB Classifier ({old_cls_path.name}): {'Found' if has_old_cls else 'Missing'}")
    print(f"2. Grayscale Classifier ({gray_cls_path.name}): {'Found' if has_gray_cls else 'Missing'}")
    print(f"3. YOLO11 Object Detector ({yolo_model_path.name}): {'Found' if has_yolo else 'Missing'}")

    m2_model = YOLO(str(m2_path)) if m2_path.exists() else None

    # Load Old RGB Classifier
    old_cls_model = None
    tf_rgb = None
    if has_old_cls:
        ckpt = torch.load(old_cls_path, map_location=device)
        old_cls_model = ResNetProvinceClassifier(n_classes=77, pretrained=False)
        if "model_state" in ckpt:
            old_cls_model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            old_cls_model.load_state_dict(ckpt, strict=False)
        old_cls_model.to(device).eval()
        tf_rgb = get_prov_transforms(is_train=False)

    # Load Grayscale Classifier
    gray_cls_model = None
    tf_gray = None
    if has_gray_cls:
        ckpt_gray = torch.load(gray_cls_path, map_location=device)
        gray_cls_model = ResNetProvinceClassifier(n_classes=77, backbone="resnet18", pretrained=False)
        if "model_state" in ckpt_gray:
            gray_cls_model.load_state_dict(ckpt_gray["model_state"], strict=False)
        else:
            gray_cls_model.load_state_dict(ckpt_gray, strict=False)
        gray_cls_model.to(device).eval()
        tf_gray = get_grayscale_val_transforms()

    # Load YOLO Detector
    yolo_model = YOLO(str(yolo_model_path)) if has_yolo else None

    test_files = sorted(list(test_img_dir.glob("*.*")))[:num_test_samples]
    print(f"\nEvaluating on {len(test_files)} independent Thai test plates...\n")

    old_cls_correct = 0
    gray_cls_correct = 0
    yolo_correct = 0

    old_cls_times = []
    gray_cls_times = []
    yolo_times = []

    for img_p in test_files:
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        with open(lbl_p) as f:
            parts = f.readline().strip().split()
            if not parts:
                continue
            gt_cls_idx = int(parts[0])
            gt_abbr = yolo_model.names[gt_cls_idx] if yolo_model else ""
            gt_name = abbr_to_name.get(gt_abbr, gt_abbr)

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        rh, rw = img_bgr.shape[:2]

        # Component extraction via Model 2
        prov_crop = None
        if m2_model is not None:
            m2_res = m2_model(img_bgr, conf=0.20, verbose=False)[0]
            for b in m2_res.boxes:
                if int(b.cls[0]) == 1:  # province
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                    prov_crop = img_bgr[by1:by2, bx1:bx2]
                    break
        if prov_crop is None or prov_crop.size == 0:
            prov_crop = img_bgr[int(rh * 0.60):, :]

        pil_c = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))

        # 1. Approach 1: Old RGB Classifier
        if old_cls_model is not None:
            t0 = time.time()
            ts_rgb = tf_rgb(pil_c).unsqueeze(0).to(device)
            with torch.no_grad():
                out = old_cls_model(ts_rgb)
                pred_idx = int(out.argmax(dim=1).item())
                pred_name = prov_map.get(str(pred_idx), "Unknown")
            old_cls_times.append((time.time() - t0) * 1000)
            if pred_name == gt_name or gt_name in pred_name:
                old_cls_correct += 1

        # 2. Approach 2: Grayscale Classifier
        if gray_cls_model is not None:
            t0 = time.time()
            ts_gray = tf_gray(pil_c).unsqueeze(0).to(device)
            with torch.no_grad():
                out = gray_cls_model(ts_gray)
                pred_idx = int(out.argmax(dim=1).item())
                pred_name = prov_map.get(str(pred_idx), "Unknown")
            gray_cls_times.append((time.time() - t0) * 1000)
            if pred_name == gt_name or gt_name in pred_name:
                gray_cls_correct += 1

        # 3. Approach 3: YOLO11 Object Detector (1-Stage Direct Plate)
        if yolo_model is not None:
            t0 = time.time()
            y_res = yolo_model(img_bgr, conf=0.15, verbose=False)[0]
            pred_yolo_abbr = None
            if len(y_res.boxes) > 0:
                best_idx = int(y_res.boxes.conf.argmax().item())
                pred_yolo_cls = int(y_res.boxes.cls[best_idx].item())
                pred_yolo_abbr = yolo_model.names[pred_yolo_cls]
            yolo_times.append((time.time() - t0) * 1000)
            if pred_yolo_abbr == gt_abbr:
                yolo_correct += 1

    total = len(test_files)
    print("=" * 80)
    print("🇹🇭 THAI BENCHMARK RESULTS SUMMARY:")
    print("=" * 80)
    if old_cls_times:
        print(f"Approach 1 (Model 2 + Old RGB Classifier):")
        print(f"  - Accuracy : {old_cls_correct}/{total} ({old_cls_correct / total * 100:.1f}%)")
        print(f"  - Latency  : {np.mean(old_cls_times):.1f} ms/image")
    if gray_cls_times:
        print(f"\nApproach 2 (Model 2 + Grayscale Classifier):")
        print(f"  - Accuracy : {gray_cls_correct}/{total} ({gray_cls_correct / total * 100:.1f}%)")
        print(f"  - Latency  : {np.mean(gray_cls_times):.1f} ms/image")
    if yolo_times:
        print(f"\nApproach 3 (YOLO11 Object Detector - 1-Stage):")
        print(f"  - Accuracy : {yolo_correct}/{total} ({yolo_correct / total * 100:.1f}%)")
        print(f"  - Latency  : {np.mean(yolo_times):.1f} ms/image")
    print("=" * 80)


def benchmark_lao(num_test_samples=50):
    print("\n" + "=" * 80)
    print("🇱🇦 Benchmarking Lao Province Recognition (3 Approaches Side-by-Side)")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    old_cls_path = PROJECT_ROOT / "weights" / "province_model_lao.pth"
    gray_cls_path = PROJECT_ROOT / "weights" / "province_model_grayscale_lao.pth"
    prov_map_path = PROJECT_ROOT / "weights" / "province_map_lao.json"
    yolo_model_path = PROJECT_ROOT / "weights" / "province_detector_yolo_lao.pt"

    test_img_dir = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_yolo_balanced" / "test" / "images"
    test_lbl_dir = PROJECT_ROOT / "datasets" / "Lao" / "lao_province_yolo_balanced" / "test" / "labels"

    if not test_img_dir.exists():
        print(f"Lao test directory not found: {test_img_dir}")
        return

    with open(prov_map_path, "r", encoding="utf-8") as f:
        prov_map = json.load(f)

    has_old_cls = old_cls_path.exists()
    has_gray_cls = gray_cls_path.exists()
    has_yolo = yolo_model_path.exists()

    print(f"1. Old RGB Classifier ({old_cls_path.name}): {'Found' if has_old_cls else 'Missing'}")
    print(f"2. Grayscale Classifier ({gray_cls_path.name}): {'Found' if has_gray_cls else 'Missing'}")
    print(f"3. YOLO11 Object Detector ({yolo_model_path.name}): {'Found' if has_yolo else 'Missing'}")

    m2_path = PROJECT_ROOT / "weights" / "component_detector.pt"
    m2_model = YOLO(str(m2_path)) if m2_path.exists() else None

    # Load Old RGB Classifier
    old_cls_model = None
    tf_rgb = None
    if has_old_cls:
        ckpt = torch.load(old_cls_path, map_location=device)
        old_cls_model = ResNetProvinceClassifier(n_classes=len(prov_map), pretrained=False)
        if "model_state" in ckpt:
            old_cls_model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            old_cls_model.load_state_dict(ckpt, strict=False)
        old_cls_model.to(device).eval()
        tf_rgb = get_prov_transforms(is_train=False)

    # Load Grayscale Classifier
    gray_cls_model = None
    tf_gray = None
    if has_gray_cls:
        ckpt_gray = torch.load(gray_cls_path, map_location=device)
        gray_cls_model = ResNetProvinceClassifier(n_classes=len(prov_map), backbone="resnet18", pretrained=False)
        if "model_state" in ckpt_gray:
            gray_cls_model.load_state_dict(ckpt_gray["model_state"], strict=False)
        else:
            gray_cls_model.load_state_dict(ckpt_gray, strict=False)
        gray_cls_model.to(device).eval()
        tf_gray = get_grayscale_val_transforms()

    # Load YOLO Detector
    yolo_model = YOLO(str(yolo_model_path)) if has_yolo else None

    test_files = sorted(list(test_img_dir.glob("*.*")))[:num_test_samples]
    print(f"\nEvaluating on {len(test_files)} independent Lao test plates...\n")

    old_cls_correct = 0
    gray_cls_correct = 0
    yolo_correct = 0

    old_cls_times = []
    gray_cls_times = []
    yolo_times = []

    for img_p in test_files:
        lbl_p = test_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        with open(lbl_p) as f:
            parts = f.readline().strip().split()
            if not parts:
                continue
            gt_cls_idx = int(parts[0])

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        rh, rw = img_bgr.shape[:2]

        # Extract province crop using Model 2 with vertical flip (user workflow)
        prov_crop = None
        if m2_model is not None:
            flipped = cv2.flip(img_bgr, 0)
            res = m2_model(flipped, conf=0.15, verbose=False)[0]
            best_conf = 0.0
            for b in res.boxes:
                c_name = m2_model.names[int(b.cls[0])].lower()
                c_conf = float(b.conf[0])
                if ("prov" in c_name or int(b.cls[0]) == 1) and c_conf > best_conf:
                    bx1, by1, bx2, by2 = b.xyxy[0].cpu().numpy().astype(int)
                    orig_y1 = max(0, rh - by2)
                    orig_y2 = min(rh, rh - by1)
                    orig_x1 = max(0, bx1)
                    orig_x2 = min(rw, bx2)
                    candidate = img_bgr[orig_y1:orig_y2, orig_x1:orig_x2]
                    if candidate.size > 0 and candidate.shape[0] >= 6 and candidate.shape[1] >= 15:
                        prov_crop = candidate
                        best_conf = c_conf

        if prov_crop is None or prov_crop.size == 0:
            prov_crop = img_bgr[int(rh * 0.02) : int(rh * 0.38), int(rw * 0.12) : int(rw * 0.88)]

        pil_c = Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB))

        # 1. Approach 1: Old RGB Classifier
        if old_cls_model is not None:
            t0 = time.time()
            ts_rgb = tf_rgb(pil_c).unsqueeze(0).to(device)
            with torch.no_grad():
                out = old_cls_model(ts_rgb)
                pred_idx = int(out.argmax(dim=1).item())
            old_cls_times.append((time.time() - t0) * 1000)
            if pred_idx == gt_cls_idx:
                old_cls_correct += 1

        # 2. Approach 2: Grayscale Classifier
        if gray_cls_model is not None:
            t0 = time.time()
            ts_gray = tf_gray(pil_c).unsqueeze(0).to(device)
            with torch.no_grad():
                out = gray_cls_model(ts_gray)
                pred_idx = int(out.argmax(dim=1).item())
            gray_cls_times.append((time.time() - t0) * 1000)
            if pred_idx == gt_cls_idx:
                gray_cls_correct += 1

        # 3. Approach 3: YOLO11 Object Detector
        if yolo_model is not None:
            t0 = time.time()
            y_res = yolo_model(img_bgr, conf=0.15, verbose=False)[0]
            pred_cls_idx = None
            if len(y_res.boxes) > 0:
                best_idx = int(y_res.boxes.conf.argmax().item())
                pred_cls_idx = int(y_res.boxes.cls[best_idx].item())
            yolo_times.append((time.time() - t0) * 1000)
            if pred_cls_idx == gt_cls_idx:
                yolo_correct += 1

    total = len(test_files)
    print("=" * 80)
    print("🇱🇦 LAO BENCHMARK RESULTS SUMMARY:")
    print("=" * 80)
    if old_cls_times:
        print(f"Approach 1 (Banner Crop + Old RGB Classifier):")
        print(f"  - Accuracy : {old_cls_correct}/{total} ({old_cls_correct / total * 100:.1f}%)")
        print(f"  - Latency  : {np.mean(old_cls_times):.1f} ms/image")
    if gray_cls_times:
        print(f"\nApproach 2 (Banner Crop + Grayscale Classifier):")
        print(f"  - Accuracy : {gray_cls_correct}/{total} ({gray_cls_correct / total * 100:.1f}%)")
        print(f"  - Latency  : {np.mean(gray_cls_times):.1f} ms/image")
    if yolo_times:
        print(f"\nApproach 3 (YOLO11 Object Detector - 1-Stage):")
        print(f"  - Accuracy : {yolo_correct}/{total} ({yolo_correct / total * 100:.1f}%)")
        print(f"  - Latency  : {np.mean(yolo_times):.1f} ms/image")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_thai()
    benchmark_lao()
