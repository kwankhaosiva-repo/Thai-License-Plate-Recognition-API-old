"""
src/test_plate_quad_visual.py

Tests PlateOBBNet ONNX (weights/plate_quad_detector_opset18.onnx) on unseen test images at 640x640.
Unwarps the tilted plates to 320x160 straight plates and generates side-by-side visual comparisons.
Automatically regenerates: output/plate_quad_verification/report.html
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = PROJECT_ROOT / "weights" / "plate_quad_detector_opset18.onnx"
TEST_IMG_DIR = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new" / "test" / "images"
OUT_DIR = PROJECT_ROOT / "output" / "plate_quad_verification"


def generate_html_report():
    imgs = sorted([f.name for f in OUT_DIR.glob("*.jpg")])
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PlateOBBNet 1-Model Rotated Bounding Box Detection & Rectification</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; text-align: center; margin-bottom: 8px; }}
        .subtitle {{ text-align: center; color: #94a3b8; margin-bottom: 32px; font-size: 1.1rem; }}
        .grid {{ display: flex; flex-direction: column; gap: 28px; max-width: 1200px; margin: 0 auto; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); border: 1px solid #334155; }}
        .card-title {{ font-size: 0.95rem; color: #38bdf8; margin-bottom: 12px; word-break: break-all; font-family: monospace; }}
        .card img {{ width: 100%; border-radius: 8px; display: block; }}
        .badge {{ background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; margin-left: 8px; }}
    </style>
</head>
<body>
    <h1>🎯 PlateOBBNet (1-Pass Rigid OBB 640x640 Detector)</h1>
    <div class="subtitle">Standalone ONNX (Opset 18) &bull; Commercially Free (BSD-3) &bull; 100% Parallel Rigid Rectangular Rectification</div>
    <div class="grid">
"""
    for img in imgs:
        html += f"""        <div class="card">
            <div class="card-title">{img} <span class="badge">1-Pass Rigid OBB Rectified</span></div>
            <img src="{img}" alt="{img}">
        </div>\n"""

    html += """    </div>
</body>
</html>
"""
    (OUT_DIR / "report.html").write_text(html, encoding="utf-8")


def test_plate_quad_visual(num_samples: int = 10, input_size: int = 640, conf_thresh: float = 0.30):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    print("=" * 70)
    print(f"🔍 Testing PlateOBBNet 1-Pass Rotated Detection & Rectification at {input_size}x{input_size}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Outputs : {OUT_DIR}")
    print("=" * 70)

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    test_images = sorted(list(TEST_IMG_DIR.glob("*.*")))[:num_samples]
    if not test_images:
        print(f"No test images found in {TEST_IMG_DIR}")
        return

    for idx, img_p in enumerate(test_images, 1):
        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue

        orig_h, orig_w = orig_img.shape[:2]

        # Preprocess to 640x640
        resized = cv2.resize(orig_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = ((rgb - mean) / std).transpose(2, 0, 1)
        tensor = np.expand_dims(normalized, axis=0).astype(np.float32)

        # 1-pass ONNX inference
        outputs = session.run(None, {"images": tensor})
        corners_batch, scores_batch = outputs  # corners: (1, 3, 4, 2), scores: (1, 3)

        top_score = float(scores_batch[0, 0])
        top_corners = corners_batch[0, 0]  # (4, 2) in normalized [0, 1]

        # Convert detected corners to original image pixel coordinates
        vis_img = orig_img.copy()
        pts_orig = []
        for c in range(4):
            px = float(top_corners[c, 0] * orig_w)
            py = float(top_corners[c, 1] * orig_h)
            pts_orig.append([px, py])

        pts_orig = np.array(pts_orig, dtype=np.float32)

        if top_score >= conf_thresh:
            # Draw polygon lines
            poly_pts = pts_orig.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=3)

            # Draw corner dots with labels
            colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # TL, TR, BR, BL
            labels = ["TL", "TR", "BR", "BL"]
            for i in range(4):
                cx, cy = int(pts_orig[i, 0]), int(pts_orig[i, 1])
                cv2.circle(vis_img, (cx, cy), 6, colors[i], -1)
                cv2.putText(vis_img, labels[i], (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

            # Unwarp plate to 320x160 straight rectangle
            dst_quad = np.array([[0, 0], [320, 0], [320, 160], [0, 160]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts_orig, dst_quad)
            unwarped = cv2.warpPerspective(orig_img, M, (320, 160))

            # Create side-by-side visualization
            vis_h = 400
            vis_w = int(orig_w * (400.0 / orig_h))
            vis_resized = cv2.resize(vis_img, (vis_w, vis_h))

            # Canvas with unwarped plate on the right
            canvas = np.full((vis_h, vis_w + 340, 3), 30, dtype=np.uint8)
            canvas[:, :vis_w] = vis_resized
            canvas[120:280, vis_w + 10:vis_w + 330] = unwarped

            cv2.putText(canvas, f"Score: {top_score*100:.1f}%", (vis_w + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            cv2.putText(canvas, "Rigid OBB Unwarped", (vis_w + 20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            out_file = OUT_DIR / f"test_{idx:02d}_{img_p.stem}_rectified.jpg"
            cv2.imwrite(str(out_file), canvas)
            print(f"[{idx:02d}/{len(test_images)}] Saved: {out_file.name} (Conf: {top_score*100:.1f}%)")
        else:
            print(f"[{idx:02d}/{len(test_images)}] Low conf ({top_score:.2f}) on {img_p.name}")

    generate_html_report()
    print("=" * 70)
    print(f"✅ Visual verification completed! Report updated at: {OUT_DIR / 'report.html'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    test_plate_quad_visual(num_samples=args.samples, input_size=args.size)
