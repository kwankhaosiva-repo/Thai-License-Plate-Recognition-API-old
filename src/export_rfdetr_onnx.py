"""src/export_rfdetr_onnx.py

Export RF-DETR (Base and Small) PyTorch checkpoints to ONNX format (opset 17).
Suitable for direct consumption in C# (Microsoft.ML.OnnxRuntime), C++, Python, or edge runtimes.
"""

import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rfdetr_export")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def export_model_to_onnx(
    checkpoint_path: Path,
    output_onnx_path: Path,
    variant: str = "base",
    opset_version: int = 17,
    dynamic_batch: bool = False,
) -> Path:
    """Export an RF-DETR checkpoint (.pt) to ONNX with opset_version.
    
    Args:
        checkpoint_path: Path to input .pt checkpoint.
        output_onnx_path: Path to target .onnx file.
        variant: 'base' or 'small'.
        opset_version: Target ONNX opset version (default: 17).
        dynamic_batch: Whether to export with dynamic batch dimension.
        
    Returns:
        Path to the exported ONNX file.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading RF-DETR ({variant}) from {checkpoint_path}...")
    if variant.lower() == "small":
        from rfdetr import RFDETRSmall
        model = RFDETRSmall.from_checkpoint(str(checkpoint_path), trust_checkpoint=True)
    else:
        from rfdetr import RFDETRBase
        model = RFDETRBase.from_checkpoint(str(checkpoint_path), trust_checkpoint=True)

    output_dir = output_onnx_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to ONNX: {output_onnx_path} (opset={opset_version})...")
    exported_file = model.export(
        output_dir=str(output_dir),
        output_name=output_onnx_path.name,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        format="onnx",
        verbose=False,
    )
    logger.info(f"Successfully exported ONNX model to: {exported_file}")
    return Path(exported_file)


def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR models to ONNX (opset 17)")
    parser.add_argument(
        "--model",
        choices=["base", "small", "both"],
        default="both",
        help="Which Model 1 plate detector to export (base, small, or both). Default: both",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Enable dynamic batch size in ONNX graph",
    )
    args = parser.parse_args()

    targets = []
    if args.model in ("base", "both"):
        targets.append((
            WEIGHTS_DIR / "plate_detector_rfdetr.pt",
            WEIGHTS_DIR / "plate_detector_rfdetr.onnx",
            "base",
        ))
    if args.model in ("small", "both"):
        targets.append((
            WEIGHTS_DIR / "plate_detector_rfdetr_small.pt",
            WEIGHTS_DIR / "plate_detector_rfdetr_small.onnx",
            "small",
        ))

    for ckpt_path, onnx_path, variant in targets:
        export_model_to_onnx(
            checkpoint_path=ckpt_path,
            output_onnx_path=onnx_path,
            variant=variant,
            opset_version=args.opset,
            dynamic_batch=args.dynamic_batch,
        )


if __name__ == "__main__":
    main()
