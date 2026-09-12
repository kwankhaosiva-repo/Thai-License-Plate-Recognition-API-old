"""
src/train_rtdetr_enterprise.py

Enterprise-grade, 100% Permissive Commercial License (Apache-2.0) Model Training Pipeline
Replaces copyleft AGPL-3.0 YOLO models with RT-DETR / OpenMMLab RTMDet-Ins.

Dataset COCO Mappings:
  - Model 1 (Plate Polygon / Instance Segmentation):
      datasets/Thai/LPR 2 - Polygon.yolov11_new/train/_annotations.coco.json
      datasets/Thai/LPR 2 - Polygon.yolov11_new/valid/_annotations.coco.json
  - Model 2 (Components: plate_char & province):
      datasets/Thai/LPR 2 - Charactor Detection.yolov11/train/_annotations.coco.json
      datasets/Thai/LPR 2 - Charactor Detection.yolov11/valid/_annotations.coco.json
  - Model 3A (Character Box Detection):
      datasets/Thai/LPR 2 - Character Box Detection.yolov11/train/_annotations.coco.json
      datasets/Thai/LPR 2 - Character Box Detection.yolov11/valid/_annotations.coco.json
  - Model 3B (Province Detection):
      datasets/Thai/thai-car-license-plate-province.v5i.yolov11/train/_annotations.coco.json
  - Lao Plate Detection:
      datasets/Lao/laos plate.v3i.yolov11/train/_annotations.coco.json
      datasets/Lao/Lao License Plates.v2i.yolov11/train/_annotations.coco.json
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Registered Dataset COCO Specifications
DATASET_REGISTRY = {
    "model_1_plate_polygon": {
        "task": "instance_segmentation",
        "recommended_arch": "RTMDet-Ins (Apache 2.0)",
        "train_json": PROJECT_ROOT / "datasets/Thai/LPR 2 - Polygon.yolov11_new/train/_annotations.coco.json",
        "train_img_dir": PROJECT_ROOT / "datasets/Thai/LPR 2 - Polygon.yolov11_new/train",
        "val_json": PROJECT_ROOT / "datasets/Thai/LPR 2 - Polygon.yolov11_new/valid/_annotations.coco.json",
        "val_img_dir": PROJECT_ROOT / "datasets/Thai/LPR 2 - Polygon.yolov11_new/valid",
        "classes": ["plate"],
    },
    "model_2_components": {
        "task": "object_detection",
        "recommended_arch": "RT-DETR / RTMDet-s (Apache 2.0)",
        "train_json": PROJECT_ROOT / "datasets/Thai/LPR 2 - Charactor Detection.yolov11/train/_annotations.coco.json",
        "train_img_dir": PROJECT_ROOT / "datasets/Thai/LPR 2 - Charactor Detection.yolov11/train",
        "val_json": PROJECT_ROOT / "datasets/Thai/LPR 2 - Charactor Detection.yolov11/valid/_annotations.coco.json",
        "val_img_dir": PROJECT_ROOT / "datasets/Thai/LPR 2 - Charactor Detection.yolov11/valid",
        "classes": ["plate_char", "province"],
    },
    "model_3a_char_box": {
        "task": "object_detection",
        "recommended_arch": "RT-DETR / RTMDet-tiny (Apache 2.0)",
        "train_json": PROJECT_ROOT / "datasets/Thai/LPR 2 - Character Box Detection.yolov11/train/_annotations.coco.json",
        "train_img_dir": PROJECT_ROOT / "datasets/Thai/LPR 2 - Character Box Detection.yolov11/train",
        "val_json": PROJECT_ROOT / "datasets/Thai/LPR 2 - Character Box Detection.yolov11/valid/_annotations.coco.json",
        "val_img_dir": PROJECT_ROOT / "datasets/Thai/LPR 2 - Character Box Detection.yolov11/valid",
        "classes": ["Each Thai Charactor car plate"],
    },
    "model_lao_plate": {
        "task": "object_detection",
        "recommended_arch": "RT-DETR / RTMDet-s (Apache 2.0)",
        "train_json": PROJECT_ROOT / "datasets/Lao/laos plate.v3i.yolov11/train/_annotations.coco.json",
        "train_img_dir": PROJECT_ROOT / "datasets/Lao/laos plate.v3i.yolov11/train",
        "val_json": PROJECT_ROOT / "datasets/Lao/laos plate.v3i.yolov11/valid/_annotations.coco.json",
        "val_img_dir": PROJECT_ROOT / "datasets/Lao/laos plate.v3i.yolov11/valid",
        "classes": ["plate"],
    },
}


def verify_coco_registry():
    print("=" * 70)
    print("--- Verifying Enterprise COCO Datasets for RT-DETR / RTMDet ---")
    print("=" * 70)
    all_ok = True
    for name, info in DATASET_REGISTRY.items():
        print(f"\n[Model Target: {name}] ({info['recommended_arch']})")
        t_json = info["train_json"]
        v_json = info["val_json"]
        if t_json.exists():
            with open(t_json, "r") as f:
                d = json.load(f)
            num_imgs = len(d.get("images", []))
            num_anns = len(d.get("annotations", []))
            print(f"  Train COCO: OK ({num_imgs} images, {num_anns} annotations)")
        else:
            print(f"  Train COCO: MISSING ({t_json})")
            all_ok = False

        if v_json.exists():
            with open(v_json, "r") as f:
                d = json.load(f)
            num_imgs = len(d.get("images", []))
            num_anns = len(d.get("annotations", []))
            print(f"  Valid COCO: OK ({num_imgs} images, {num_anns} annotations)")
        else:
            print(f"  Valid COCO: MISSING ({v_json})")
            all_ok = False

    return all_ok


def generate_mmdetection_config(target_name: str) -> str:
    """Generates an Apache-2.0 OpenMMLab MMDetection config string for a target model."""
    if target_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown target: {target_name}")

    info = DATASET_REGISTRY[target_name]
    classes_str = str(tuple(info["classes"])) if len(info["classes"]) > 1 else f"('{info['classes'][0]}',)"

    if info["task"] == "instance_segmentation":
        base_config = "rtmdet-ins_s_8xb32-300e_coco"
    else:
        base_config = "rtmdet_s_8xb32-300e_coco"

    cfg_str = f"""# Auto-generated MMDetection (Apache 2.0) Config for {target_name}
_base_ = 'configs/rtmdet/{base_config}.py'

metainfo = dict(classes={classes_str})

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root='{info["train_img_dir"]}',
        ann_file='_annotations.coco.json',
        data_prefix=dict(img=''),
    )
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    dataset=dict(
        metainfo=metainfo,
        data_root='{info["val_img_dir"]}',
        ann_file='_annotations.coco.json',
        data_prefix=dict(img=''),
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file='{info["val_json"]}',
    metric=['bbox' if '{info["task"]}' == 'object_detection' else 'segm']
)
test_evaluator = val_evaluator

max_epochs = 40
train_cfg = dict(max_epochs=max_epochs, val_interval=2)
"""
    return cfg_str


if __name__ == "__main__":
    verify_coco_registry()
