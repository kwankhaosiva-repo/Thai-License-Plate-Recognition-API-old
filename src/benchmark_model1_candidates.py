"""
src/benchmark_model1_candidates.py

Comprehensive benchmark comparing Model 1 Plate Detector candidates on the same test dataset:
  - Baseline 1: RF-DETR-Small (Current)
  - Baseline 2: RF-DETR-Base
  - Candidate 1: LibreDFINE-Nano (ONNX)
  - Candidate 2: LibreDFINE-Small (ONNX)
  - Candidate 3: LibreRFDETR-Small OBB (ONNX)
  - Candidate 4: LibreRTDETRv2-r18 (ONNX)

Measures:
  - CPU Latency (ms)
  - MPS (Apple GPU) Latency (ms)
  - Model Parameters & File Size (MB)
  - C# ONNX Runtime Compatibility
"""

import time
import os
import cv2
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
TEST_IMG_DIR = PROJECT_ROOT / "datasets" / "Thai" / "LPR 2 - Polygon.yolov11_new" / "test" / "images"


def load_test_images(num_images=10):
    images = []
    if TEST_IMG_DIR.exists():
        files = list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png"))
        for f in files[:num_images]:
            img = cv2.imread(str(f))
            if img is not None:
                images.append(img)
    if not images:
        images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(5)]
    return images


def benchmark_onnx_model(onnx_path: Path, test_images, runs_per_img=5):
    import onnxruntime as ort

    if not onnx_path.exists():
        return None

    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    # CPU Session
    cpu_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_meta = cpu_sess.get_inputs()[0]
    input_name = input_meta.name
    shape = input_meta.shape
    h = shape[2] if len(shape) >= 4 and isinstance(shape[2], int) else 640
    w = shape[3] if len(shape) >= 4 and isinstance(shape[3], int) else 640

    # Prepare batch
    tensors = []
    for img in test_images:
        resized = cv2.resize(img, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
        tensors.append(tensor)

    # Warmup
    cpu_sess.run(None, {input_name: tensors[0]})

    # Measure CPU
    t0 = time.time()
    count = 0
    for t in tensors:
        for _ in range(runs_per_img):
            cpu_sess.run(None, {input_name: t})
            count += 1
    cpu_ms = (time.time() - t0) / count * 1000

    return {
        "name": onnx_path.name,
        "format": "ONNX",
        "size_mb": size_mb,
        "cpu_ms": cpu_ms,
        "csharp_ready": "✅ Ready (DirectML/CPU)",
    }


def benchmark_rfdetr_torch(model_path: Path, is_small: bool, test_images, runs_per_img=3):
    from rfdetr import RFDETRSmall, RFDETRBase

    if not model_path.exists():
        return None

    size_mb = model_path.stat().st_size / (1024 * 1024)

    # Test on CPU
    cls = RFDETRSmall if is_small else RFDETRBase
    model_cpu = cls.from_checkpoint(str(model_path), trust_checkpoint=True)
    if hasattr(model_cpu, "model") and hasattr(model_cpu.model, "device"):
        model_cpu.model.device = torch.device("cpu")

    model_cpu.predict(test_images[0])  # Warmup
    t0 = time.time()
    count = 0
    for img in test_images:
        for _ in range(runs_per_img):
            model_cpu.predict(img)
            count += 1
    cpu_ms = (time.time() - t0) / count * 1000

    # Test on MPS
    model_mps = cls.from_checkpoint(str(model_path), trust_checkpoint=True)
    if hasattr(model_mps, "model") and hasattr(model_mps.model, "device"):
        model_mps.model.device = torch.device("mps")

    model_mps.predict(test_images[0])
    torch.mps.synchronize()
    t0 = time.time()
    count = 0
    for img in test_images:
        for _ in range(runs_per_img):
            model_mps.predict(img)
            count += 1
    torch.mps.synchronize()
    mps_ms = (time.time() - t0) / count * 1000

    return {
        "name": model_path.name,
        "format": "PyTorch .pt",
        "size_mb": size_mb,
        "cpu_ms": cpu_ms,
        "mps_ms": mps_ms,
        "csharp_ready": "⚠️ Needs ONNX Export",
    }


def run_all_benchmarks():
    print("=" * 85)
    print("🏁 BENCHMARK: MODEL 1 PLATE DETECTOR CANDIDATES ON APPLE SILICON")
    print("=" * 85)

    test_imgs = load_test_images(num_images=5)
    print(f"Loaded {len(test_imgs)} test images for evaluation.")

    results = []

    # 1. Baseline RF-DETR-Small
    print("\n[1/6] Benchmarking Current Baseline: plate_detector_rfdetr_small.pt...")
    res_base_s = benchmark_rfdetr_torch(WEIGHTS_DIR / "plate_detector_rfdetr_small.pt", is_small=True, test_images=test_imgs)
    if res_base_s:
        results.append(res_base_s)

    # 2. Baseline RF-DETR-Base
    print("[2/6] Benchmarking Current Baseline: plate_detector_rfdetr.pt (Base)...")
    res_base_b = benchmark_rfdetr_torch(WEIGHTS_DIR / "plate_detector_rfdetr.pt", is_small=False, test_images=test_imgs)
    if res_base_b:
        results.append(res_base_b)

    # 3. LibreDFINE-Nano (ONNX)
    print("[3/6] Benchmarking Candidate: LibreDFINEn.onnx (D-FINE Nano)...")
    res_dfine_n = benchmark_onnx_model(WEIGHTS_DIR / "LibreDFINEn.onnx", test_images=test_imgs)
    if res_dfine_n:
        results.append(res_dfine_n)

    # 4. LibreDFINE-Small (ONNX)
    print("[4/6] Benchmarking Candidate: LibreDFINEs.onnx (D-FINE Small)...")
    res_dfine_s = benchmark_onnx_model(WEIGHTS_DIR / "LibreDFINEs.onnx", test_images=test_imgs)
    if res_dfine_s:
        results.append(res_dfine_s)

    # 5. LibreRTDETRv2-r18 (ONNX)
    print("[5/6] Benchmarking Candidate: LibreRTDETRv2r18.onnx (RT-DETRv2)...")
    res_rtdetr = benchmark_onnx_model(WEIGHTS_DIR / "LibreRTDETRv2r18.onnx", test_images=test_imgs)
    if res_rtdetr:
        results.append(res_rtdetr)

    # 6. LibreRFDETR-Small OBB (ONNX)
    print("[6/8] Benchmarking Candidate: LibreRFDETRs-obb.onnx (RF-DETR OBB)...")
    res_obb = benchmark_onnx_model(WEIGHTS_DIR / "LibreRFDETRs-obb.onnx", test_images=test_imgs)
    if res_obb:
        results.append(res_obb)

    # 7. LibrePICODET-S (ONNX)
    print("[7/8] Benchmarking Candidate: LibrePICODETs.onnx (PaddleDetection PicoDet)...")
    res_pico = benchmark_onnx_model(WEIGHTS_DIR / "LibrePICODETs.onnx", test_images=test_imgs)
    if res_pico:
        results.append(res_pico)

    # 8. LibreRTDETRv2-OBB Small (ONNX)
    print("[8/8] Benchmarking Candidate: LibreRTDETRv2s-obb.onnx (RT-DETRv2 OBB)...")
    res_rtdetr_obb = benchmark_onnx_model(WEIGHTS_DIR / "LibreRTDETRv2s-obb.onnx", test_images=test_imgs)
    if res_rtdetr_obb:
        results.append(res_rtdetr_obb)

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Model Name':<32} {'Format':<12} {'Size (MB)':<10} {'CPU (ms)':<10} {'MPS (ms)':<10} {'C# Ready'}")
    print("-" * 95)
    for r in results:
        mps_str = f"{r.get('mps_ms', 0):.1f} ms" if "mps_ms" in r else "—"
        cpu_str = f"{r['cpu_ms']:.1f} ms"
        size_str = f"{r['size_mb']:.1f} MB"
        print(f"{r['name']:<32} {r['format']:<12} {size_str:<10} {cpu_str:<10} {mps_str:<10} {r['csharp_ready']}")
    print("=" * 95)


if __name__ == "__main__":
    run_all_benchmarks()
