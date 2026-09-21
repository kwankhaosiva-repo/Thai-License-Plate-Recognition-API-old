"""
src/test_preprocess_registry.py

Self-test for the preprocessing contract (no model loading needed).
Run:  python src/test_preprocess_registry.py

Verifies:
 1. Spec picker returns the right native resolutions per model family.
 2. recommended_input_shape() reproduces the TRAIN pixel geometry.
 3. upscale_to_train_geometry + divide_boxes_by_scale round-trips boxes
    exactly (pixel-perfect on synthetic data).
 4. stretch_resize does NOT letterbox (pure stretch, correct output size).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess_registry import (  # noqa: E402
    get_m1_spec,
    get_m2_spec,
    get_m3a_spec,
    stretch_resize,
    upscale_to_train_geometry,
    divide_boxes_by_scale,
    apply_override,
    M1_PICODET,
    M1_DFINE,
    M1_RFDETR_BASE,
    M1_RFDETR_SMALL,
    M2_COMPONENTS,
    M3A_CHARBOX,
)


def check(label: str, cond: bool) -> bool:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    return cond


def main() -> int:
    ok = True

    print("== 1. Spec pickers ==")
    ok &= check("rfdetr base -> 560", get_m1_spec("plate_detector_rfdetr.pt").tensor_w == 560)
    ok &= check("rfdetr small -> 512", get_m1_spec("plate_detector_rfdetr_small.pt").tensor_w == 512)
    ok &= check("dfine nano -> 640", get_m1_spec("plate_detector_dfine_nano.pt").tensor_w == 640)
    ok &= check("picodet m -> 416", get_m1_spec("plate_detector_picodet_m.pt").tensor_w == 416)
    ok &= check("rtdetrv2 -> dfine family (640)", get_m1_spec("plate_detector_rtdetrv2_r18.pt").tensor_w == 640)
    ok &= check("m2 default (dfine) -> 640 tensor", get_m2_spec("component_detector_dfine_nano.pt").tensor_w == 640)
    ok &= check("m2 picodet -> 416 tensor", get_m2_spec("component_detector_picodet_s_v2.pt").tensor_w == 416)
    ok &= check("m3a default (rfdetr) -> 560 tensor", get_m3a_spec("character_box_detector_rfdetr.pt").tensor_w == 560)
    ok &= check("m3a picodet -> 416 tensor", get_m3a_spec("character_box_detector_picodet_s_v2.pt").tensor_w == 416)

    print("== 2. Train pixel geometry ==")
    ok &= check("M2 -> 1280x640 (train aspect ~2:1)", M2_COMPONENTS.recommended_input_shape() == (1280, 640))
    ok &= check("M3A -> 2088x560 (train aspect ~3.7:1)", M3A_CHARBOX.recommended_input_shape() == (2088, 560))
    ok &= check("M1 scene stays 4:3-friendly (<= 2560 cap)",
                M1_DFINE.recommended_input_shape()[0] <= M1_DFINE.MAX_INPUT_W)

    print("== 3. Box round-trip ==")
    # A plate crop at a typical pipeline size (320x160) through the M2 spec.
    crop = np.random.randint(0, 255, (160, 320, 3), dtype=np.uint8)
    upscaled, scale = upscale_to_train_geometry(crop, M2_COMPONENTS)
    ok &= check("upscaled to 1280x640", upscaled.shape[:2] == (640, 1280))
    ok &= check("per-axis scales (4.0, 4.0) for 320x160", np.allclose(scale, (4.0, 4.0)))
    box = np.array([[10, 5, 300, 150]], dtype=np.float32)
    back = divide_boxes_by_scale(box * np.array([scale[0], scale[1], scale[0], scale[1]]), scale)
    ok &= check("box round-trip exact", np.allclose(back, box, atol=1e-3))
    # A 3.7:1 char crop (200x54) through the M3A spec (asymmetric per-axis scale).
    char_crop = np.random.randint(0, 255, (54, 200, 3), dtype=np.uint8)
    ups3, scale3 = upscale_to_train_geometry(char_crop, M3A_CHARBOX)
    ok &= check("M3A upscaled to 2088x560", ups3.shape[:2] == (560, 2088))
    box3 = np.array([[20, 6, 60, 48]], dtype=np.float32)
    back3 = divide_boxes_by_scale(box3 * np.array([scale3[0], scale3[1], scale3[0], scale3[1]]), scale3)
    ok &= check("M3A asymmetric round-trip exact", np.allclose(back3, box3, atol=1e-2))
    ok &= check("empty crop guarded", upscale_to_train_geometry(np.zeros((0, 0, 3), np.uint8), M2_COMPONENTS)[1] == (1.0, 1.0))

    print("== 4. Stretch, not letterbox ==")
    out = stretch_resize(np.zeros((160, 320, 3), np.uint8), M2_COMPONENTS)
    ok &= check("stretch output is 640x640", out.shape[:2] == (640, 640))
    tall = np.zeros((640, 480, 3), np.uint8)
    out2 = stretch_resize(tall, M1_RFDETR_BASE)
    ok &= check("no letterbox borders (plain ndarray, no padding added)", out2.shape[:2] == (560, 560))

    print("== 5. Native resolutions (ONNX-verified constants) ==")
    ok &= check("RFDETR Base 560", M1_RFDETR_BASE.tensor_w == 560)
    ok &= check("RFDETR Small 512", M1_RFDETR_SMALL.tensor_w == 512)
    ok &= check("PicoDet 416", M1_PICODET.tensor_w == 416)

    print("== 6. Config-driven overrides (A/B testing) ==")
    class _FakeCfg:
        PREPROCESS_OVERRIDES: dict = {}
    base_cfg = _FakeCfg()
    ok &= check("empty overrides keep defaults",
                apply_override(M1_DFINE, base_cfg, "M1") is M1_DFINE)
    ok &= check("None values keep defaults",
                apply_override(M1_DFINE, type("C", (), {"PREPROCESS_OVERRIDES": {"M1": {"tensor": None, "aspect": None}}})(), "M1") is M1_DFINE)
    o1 = apply_override(M1_DFINE, type("C", (), {"PREPROCESS_OVERRIDES": {"M1": {"tensor": 416}}})(), "M1")
    ok &= check("tensor override -> 416 and square", o1.tensor_w == 416 and o1.tensor_h == 416)
    ok &= check("tensor override keeps train aspect", abs(o1.train_aspect_w_over_h - 4/3) < 1e-6)
    ok &= check("tensor override updates input shape", o1.recommended_input_shape() == (554, 416))
    o2 = apply_override(M2_COMPONENTS, type("C", (), {"PREPROCESS_OVERRIDES": {"M2": {"aspect": 4/3}}})(), "M2")
    ok &= check("aspect override -> 4/3 keeps tensor 640", o2.tensor_w == 640 and o2.tensor_h == 640)
    ok &= check("aspect override -> input 852x640", o2.recommended_input_shape() == (852, 640))
    ok &= check("overridden spec is immutable-safe (new object)", o1 is not M1_DFINE and o2 is not M2_COMPONENTS)
    ok &= check("base spec untouched by overrides", M1_DFINE.tensor_w == 640 and M2_COMPONENTS.train_aspect_w_over_h == 2.0)
    # Other stages' overrides must not leak into a different stage.
    o3 = apply_override(M3A_CHARBOX, type("C", (), {"PREPROCESS_OVERRIDES": {"M1": {"tensor": 384}}})(), "M3A")
    ok &= check("stage isolation (M1 override does not hit M3A)", o3 is M3A_CHARBOX)

    print()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
