"""
src/test_dfine_nano_onnx.py

Loads the fine-tuned 'weights/plate_detector_dfine_nano.onnx' using ONNX Runtime,
runs inference on test images, prints confidence & latency, and saves annotated
images to 'debug_api/dfine_nano_test_results/'.
"""

import glob
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = PROJECT_ROOT / "weights" / "plate_detector_dfine_nano.onnx"
TEST_IMG_DIR = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new" / "test" / "images"
OUT_DIR = PROJECT_ROOT / "debug_api" / "dfine_nano_test_results"


def test_dfine_nano(num_images=5, conf_thresh=0.35):
    print("=" * 70)
    print("🔍 Testing Fine-Tuned LibreDFINE-Nano (ONNX Runtime / CPU)")
    print(f"   Model  : {ONNX_PATH.name} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"   Output : {OUT_DIR}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load ONNX model
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ONNX_PATH), opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 2. Get test images
    img_files = sorted(glob.glob(str(TEST_IMG_DIR / "*.*")))[:num_images]
    if not img_files:
        print("No test images found!")
        return

    # Warmup
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    sess.run(None, {input_name: dummy})

    latencies = []
    for idx, img_path in enumerate(img_files):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h_orig, w_orig = img_bgr.shape[:2]

        # Preprocess: Resize to 640x640, BGR->RGB, Normalize [0, 1]
        resized = cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        # Run inference
        t0 = time.perf_counter()
        out = sess.run(None, {input_name: tensor})
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        # Parse outputs:
        # out[0] = pred_logits: [1, 300, 1]
        # out[1] = pred_boxes:  [1, 300, 4] in (cx, cy, w, h) normalized format
        logits = out[0][0, :, 0]
        scores = 1.0 / (1.0 + np.exp(-logits))
        boxes = out[1][0]

        keep_idx = np.where(scores >= conf_thresh)[0]
        annotated = img_bgr.copy()

        for k in keep_idx:
            cx, cy, w, h = boxes[k]
            score = scores[k]

            x1 = int((cx - w / 2) * w_orig)
            y1 = int((cy - h / 2) * h_orig)
            x2 = int((cx + w / 2) * w_orig)
            y2 = int((cy + h / 2) * h_orig)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig - 1, x2), min(h_orig - 1, y2)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"Plate: {score * 100:.1f}% ({latency_ms:.1f}ms)",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        out_file = OUT_DIR / f"result_{idx + 1}_{Path(img_path).name}"
        cv2.imwrite(str(out_file), annotated)
        print(f"[{idx + 1}/{len(img_files)}] {Path(img_path).name:<35} | Latency: {latency_ms:5.1f} ms | Plates: {len(keep_idx)}")

    print("-" * 70)
    print(f"⚡ Average CPU Latency: {np.mean(latencies):.2f} ms")
    print(f"💾 Results saved in   : {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    test_dfine_nano()
