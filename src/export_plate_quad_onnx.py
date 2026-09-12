"""
src/export_plate_quad_onnx.py

Exports trained PlateQuadNet (PyTorch checkpoint) to a standalone ONNX model
with opset_version=18 for C# / OpenCvSharp / Microsoft.ML.OnnxRuntime deployment.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None
import numpy as np

from src.models_plate_quad import PlateQuadNet, PlateQuadNetDeploy

WEIGHTS_DIR = PROJECT_ROOT / "weights"
PTH_PATH = WEIGHTS_DIR / "plate_quad_detector.pth"
ONNX_PATH = WEIGHTS_DIR / "plate_quad_detector_opset18.onnx"


def export_plate_quad_onnx(
    pth_path: Path = PTH_PATH,
    onnx_path: Path = ONNX_PATH,
    opset: int = 18,
    target_size: int = 640,
    topk: int = 3,
):
    print("=" * 70)
    print("📦 Exporting PlateQuadNet to Standalone ONNX (Opset 18)")
    print(f"Source Checkpoint : {pth_path}")
    print(f"Target ONNX       : {onnx_path}")
    print(f"Opset Version     : {opset}")
    print(f"Input Resolution  : {target_size}x{target_size}")
    print(f"Top-K Detections  : {topk}")
    print("=" * 70)

    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {pth_path}")

    # Load checkpoint
    ckpt = torch.load(pth_path, map_location="cpu")
    core_model = PlateQuadNet(pretrained=False)
    core_model.load_state_dict(ckpt["model_state_dict"])
    core_model.eval()

    # Wrap with ONNX self-decoding wrapper
    deploy_model = PlateQuadNetDeploy(core_model, topk=topk)
    deploy_model.eval()

    dummy_input = torch.randn(1, 3, target_size, target_size, dtype=torch.float32)

    # Dynamic axes: Allow variable batch size
    dynamic_axes = {
        "images": {0: "batch"},
        "corners": {0: "batch"},
        "scores": {0: "batch"},
    }

    print("\n[1/3] Running torch.onnx.export()...")
    torch.onnx.export(
        deploy_model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["corners", "scores"],
        dynamic_axes=dynamic_axes,
    )

    # Embed all weights into a single standalone ONNX file (no external .data file)
    print("\n[2/3] Validating ONNX graph & embedding all weights into single file...")
    if onnx is not None:
        from onnx.external_data_helper import load_external_data_for_model
        model_proto = onnx.load(str(onnx_path))
        load_external_data_for_model(model_proto, str(onnx_path.parent))
        onnx.save(model_proto, str(onnx_path), save_as_external_data=False)
        onnx.checker.check_model(model_proto)

        # Remove orphan .data file if created by torch.onnx.export
        data_file = onnx_path.with_name(f"{onnx_path.name}.data")
        if data_file.exists():
            data_file.unlink()

        final_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"  --> ONNX checker passed! Single standalone model: {final_size_mb:.2f} MB")

    # Test inference with ONNX Runtime
    print("\n[3/3] Testing inference with ONNX Runtime...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    test_input = np.random.randn(1, 3, target_size, target_size).astype(np.float32)
    outputs = session.run(None, {"images": test_input})

    corners, scores = outputs
    print(f"  --> Inference output 'corners' shape: {corners.shape} (Expected: (1, {topk}, 4, 2))")
    print(f"  --> Inference output 'scores'  shape: {scores.shape} (Expected: (1, {topk}))")
    print("\n" + "=" * 70)
    print(f"✅ Export Complete! Standalone ONNX ready at: {onnx_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", type=str, default=str(PTH_PATH))
    parser.add_argument("--out", type=str, default=str(ONNX_PATH))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    export_plate_quad_onnx(
        pth_path=Path(args.pth),
        onnx_path=Path(args.out),
        opset=args.opset,
        target_size=args.size,
        topk=args.topk,
    )
