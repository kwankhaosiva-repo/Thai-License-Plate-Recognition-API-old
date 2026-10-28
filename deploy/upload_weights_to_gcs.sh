#!/usr/bin/env bash
# ============================================================================
# deploy/upload_weights_to_gcs.sh
#
# Uploads the EXACT production runtime file set to the GCS weights bucket.
# The container (see Dockerfile CMD) runs src/download_weights.py at boot,
# which downloads every blob in this bucket into weights/ — so this list is
# the single source of truth for what production loads.
#
# Usage:
#   bash deploy/upload_weights_to_gcs.sh                  # default bucket/project
#   GCS_BUCKET=my-bucket GCP_PROJECT=my-project bash deploy/upload_weights_to_gcs.sh
#
# NOTE: Keep this list in sync with src/config.py EASY CONFIG:
#   MODEL_1  = plate_detector_picodet_s_v2
#   MODEL_2  = component_detector_picodet_s_v2
#   MODEL_3A = character_box_detector_dfine_nano_v2   (dfine_nano beats picodet here)
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/weights"
GCS_BUCKET="${GCS_BUCKET:-lpr-weight}"
GCP_PROJECT="${GCP_PROJECT:-lpr-car-plate}"

FILES=(
  # --- Model 1: Plate Detector & 4-Corner Warp Regressor ---
  plate_detector_picodet_s_v6pico320.pt
  plate_detector_picodet_s_v6pico320.onnx
  plate_corner_regressor.pth
  plate_corner_regressor.onnx

  # --- Model 1.5: Country Classifier ---
  country_classifier.pth
  country_classifier.onnx

  # --- Model 2: Component Layout Detector ---
  component_detector_picodet_s_v6pico320.pt

  # --- Model 3A: Thai Character Box & Recognition ---
  character_box_detector_picodet_s_v6pico320.pt
  character_box_detector_picodet_s_v6pico320.onnx
  character_classifier_v6pico320.pth
  ocr_model.pth
  ocr_model_v3.pth
  digit_classifier.pth

  # --- Model 3B: Thai Province Model ---
  province_model_grayscale_thai_v6pico320.pth

  # --- Lao Pipeline ---
  plate_detector_lao_dfine_nano.pt
  character_box_detector_picodet_s_lao_m3a_v6.onnx
  character_recognition_mobilenetv4_lao_v7.onnx
  province_detector_picodet_s_lao_v6.onnx
  province_model_grayscale_lao_v6.pth
  province_model_grayscale_lao_v6.onnx

  # --- JSON class maps ---
  char_classifier_map.json
  char_classifier_map_lao.json
  char_classifier_map_lao_v7.json
  int_to_char.json
  int_to_char_lao.json
  province_map.json
  province_map_lao.json
  province_abbr_map.json
)

echo "Project : $GCP_PROJECT"
echo "Bucket  : gs://$GCS_BUCKET"
echo "Files   : ${#FILES[@]}"
echo

missing=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$WEIGHTS_DIR/$f" ]]; then
    echo "❌ MISSING locally: weights/$f"
    missing=1
  fi
done
if [[ $missing -ne 0 ]]; then
  echo "Abort — train/export the missing files first."
  exit 1
fi

for f in "${FILES[@]}"; do
  size=$(du -h "$WEIGHTS_DIR/$f" | cut -f1)
  echo "⬆️  $f ($size)"
  gsutil -q cp "$WEIGHTS_DIR/$f" "gs://$GCS_BUCKET/$f"
done

echo
echo "✅ Uploaded ${#FILES[@]} files to gs://$GCS_BUCKET"
echo "   Verify: gsutil ls -l gs://$GCS_BUCKET | wc -l"
