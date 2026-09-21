import os
import torch 
from pathlib import Path

class Config:
    # ============================================================
    # EASY CONFIG — Edit these settings to switch device & models
    # ============================================================

    # --- Compute Device ---
    # True  = Force CPU only (safe for any machine, ~150 ms on Apple Silicon)
    # False = Auto-detect: CUDA → MPS → CPU
    FORCE_CPU = True

    # --- Model 1: Plate Detector ---
    # Options (fastest → most accurate / specialized):
    #   "plate_detector_picodet_s.pt"        ← PicoDet-S (⚡ Ultra-fast: ~6.5 ms CPU, 3.8 MB ONNX, Apache-2.0)
    #   "plate_detector_picodet_m.pt"        ← PicoDet-M (⚡ Medium: ~12 ms CPU, 8.9 MB ONNX, Apache-2.0)
    #   "plate_detector_mobilenet_v3.pt"     ← MobileNetV3 Detector (Ultra-lightweight baseline)
    #   "plate_detector_dfine_nano.pt"       ← D-FINE-Nano (⚡ Recommended: ~27 ms CPU, 15 MB ONNX, MIT)
    #   "plate_detector_dfine_small.pt"      ← D-FINE-Small (High accuracy: ~63 ms CPU, 40 MB ONNX, MIT)
    #   "plate_detector_rfdetr_nano.pt"      ← RF-DETR-Nano (⚡ Ultra-fast transformer, Apache-2.0)
    #   "plate_detector_rfdetr_small.pt"     ← RF-DETR-Small (Transformer baseline: ~84 ms CPU, Apache-2.0)
    #   "plate_detector_rfdetr_obb_small.pt" ← RF-DETR-OBB Small (1-Stage Rotated/Angled: Apache-2.0/MIT)
    #   "plate_detector_rtdetrv2_r18.pt"     ← RT-DETRv2-R18 (Apache-2.0)
    #   "plate_detector_rfdetr.pt"           ← RF-DETR-Base (More accurate, heavier)
    #   "plate_detector_rtdetr.pt"           ← RT-DETR-L (Older baseline)
    MODEL_1_FILENAME = "plate_detector_picodet_s_v2.pt"

    # --- Model 2: Component Detector (plate_char / province bbox) ---
    #   "component_detector_dfine_nano.pt"     ← D-FINE-Nano (⚡ recommended: MIT, ultra-fast & high mAP)
    #   "component_detector_rfdetr_nano.pt"    ← RF-DETR-Nano (Apache-2.0)
    #   "component_detector_rfdetr_small.pt"   ← RF-DETR-Small
    #   "component_detector_rfdetr.pt"         ← RF-DETR-Base
    #   "component_detector_rtdetr.pt"         ← RT-DETR-L (older)
    MODEL_2_FILENAME = "component_detector_picodet_s_v2.pt"

    # --- Model 3A: Character Box Detector ---
    #   "character_box_detector_picodet_s_v2.pt" ← PicoDet-S v2 (⚡ BEST measured on val: P=0.81/R=0.93 with NMS, dup~0%)
    #   "character_box_detector_picodet_m_v2.pt" ← PicoDet-M v2
    #   "character_box_detector_dfine_nano_v2.pt" ← D-FINE-Nano v2 (highest recall 0.99, needs NMS — see api_server)
    #   "character_box_detector_dfine_small_v2.pt"← D-FINE-Small v2 (recall 0.98 with NMS)
    #   "character_box_detector_dfine_nano.pt"  ← D-FINE-Nano v1 (⚡ recommended: MIT, precise char localization)
    #   "character_box_detector_dfine_small.pt" ← D-FINE-Small v1 (MIT, higher capacity)
    #   "character_box_detector_rfdetr_nano.pt" ← RF-DETR-Nano (Apache-2.0)
    #   "character_box_detector_rfdetr_small.pt"← RF-DETR-Small
    #   "character_box_detector_rfdetr.pt"      ← RF-DETR-Base
    #   "character_box_detector_rtdetr.pt"      ← RT-DETR-L (older)
    # NOTE: filename must be the EXACT weight filename (no "...pt_v2.pt" mistakes —
    # a nonexistent name silently falls back to an older checkpoint via ACTIVE_CHAR_BOX_MODEL_PATH).
    MODEL_3A_FILENAME = "character_box_detector_dfine_nano_v2.pt"

    # --- Model 3A: Character Classifier (MobileNetV2) ---
    #   "character_classifier.pth" ← MobileNetV2 (50 Thai character & digit classes)
    CHAR_CLASSIFIER_THAI_FILENAME = "character_classifier.pth"

    # --- Model 3A: OCR (CTC Text Recognition) ---
    OCR_FILENAME = "ocr_model.pth"

    # --- Model 3B: Thai Province Classifier ---
    #   "province_model_resnet34_grayscale_thai.pth" ← Grayscale ResNet34 (recommended: 256x80, handles accents)
    #   "province_model_grayscale_thai.pth"          ← Grayscale ResNet18 (legacy: 256x64)
    #   "province_model.pth"                         ← ResNet34 (Color)
    MODEL_3B_THAI_FILENAME = "province_model_grayscale_thai.pth"

    # --- Lao Plate Detector ---
    #   "plate_detector_lao_dfine_nano.pt"   ← D-FINE-Nano (⚡ recommended: MIT)
    #   "plate_detector_lao_dfine_small.pt"  ← D-FINE-Small (Higher mAP / robust: MIT)
    #   "plate_detector_lao_rfdetr_nano.pt"  ← RF-DETR-Nano (Apache-2.0)
    #   "plate_detector_lao_rfdetr_small.pt" ← RF-DETR-Small
    #   "plate_detector_lao_rfdetr.pt"       ← RF-DETR-Base
    MODEL_LAO_FILENAME = "plate_detector_lao_dfine_nano.pt"

    # --- Model 3B: Lao Province Classifier ---
    MODEL_3B_LAO_FILENAME = "province_model_grayscale_lao.pth"

    # --- Real-Time Stream & Video Configuration ---
    # Model 1 Plate Confidence Threshold for real-time video streams and video uploads:
    # Set to 0.80 for sharp, high-confidence detection and clean rejection of background clutter.
    STREAM_CONF_M1 = 0.80
    VIDEO_CONF_M1 = 0.80
    STREAM_MIN_VEHICLE_AREA = 1800
    STREAM_TARGET_SAMPLES = 1     # Immediate lock-on on valid detection, then aggregates up to 5 frames
    STREAM_MAX_SESSION_FRAMES = 5 # Maximum frames to process per vehicle session (caps compute, averages 5 diverse frames)
    STREAM_FRAME_SKIP = 2         # Skip only 1 frame for higher frequency detection during car movement (captures passing cars reliably)
    STREAM_SESSION_SEC = 5.0      # 5-second tracking & aggregation window: accumulates and averages frames of the same plate into 1 result
    STREAM_COOLDOWN_SEC = 5.0     # 5-second debounce window to prevent duplicate records for the same passing vehicle

    # --- Google Cloud Platform (Firestore & BigQuery) Configuration ---
    GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "lpr-car-plate")
    GCP_KEY_FILENAME = os.environ.get("GCP_KEY_FILENAME", "lpr-car-plate-e4c1f71338f3.json")
    FIRESTORE_DATABASE_ID = os.environ.get("FIRESTORE_DATABASE_ID", "lpr-db")
    FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "recognition_history")
    BIGQUERY_DATASET = os.environ.get("BIGQUERY_DATASET", "lpr_query")
    BIGQUERY_TABLE = os.environ.get("BIGQUERY_TABLE", "lpr_history")
    GCP_CLOUD_SYNC_ENABLED = os.environ.get("GCP_CLOUD_SYNC_ENABLED", "1") == "1"

    # ============================================================
    # SERVE / PRE-PROCESS GEOMETRY — READ THIS BEFORE A/B TESTING
    # ============================================================
    # Every detector here was trained with PURE STRETCH resize to a square
    # tensor — none of them were trained with background padding. The pipeline
    # (src/preprocess_registry.py + api_server.py) reproduces the TRAIN pixel
    # geometry at serve time automatically.
    #
    # PER-MODEL SERVE SPEC (single source of truth: preprocess_registry.py):
    #
    #   Stage / model                      Tensor (stretch)  Input aspect fed              Norm       Predict imgsz
    #   ---------------------------------  ----------------  ----------------------------  ---------  -------------
    #   M1  RF-DETR-Base                   560 x 560         4:3 scene (1.33)              ImageNet   n/a (fixed native)
    #   M1  RF-DETR-Small                  512 x 512         4:3                           ImageNet   n/a
    #   M1  RF-DETR-Nano                   384 x 384         4:3                           ImageNet   n/a
    #   M1  D-FINE Nano/Small (LibreYOLO)  640 x 640         4:3                           /255       640
    #   M1  PicoDet-S/M (LibreYOLO)        416 x 416         4:3                           /255       416  (NOT 640!)
    #   M1  YOLO11 / RT-DETR (Ultralytics) 640 x 640         4:3 letterbox (grey pad)      /255       640/1280
    #   M1.5  Country cls (MobileNetV3)    256 x 128 (WxH)   stretch from 320x160 plate    ImageNet   n/a
    #   M2  D-FINE Nano components         640 x 640         crop upscaled to 1280x640 (2:1)  /255     640
    #   M2  PicoDet-S/M components         416 x 416         crop upscaled to 832x416 (2:1)   /255     416
    #   M2  RF-DETR-Base components        560 x 560         same 2:1 upscale              ImageNet   n/a
    #   M3A RF-DETR-Base char box          560 x 560         crop upscaled to 2088x560 (3.7:1) ImageNet n/a
    #   M3A D-FINE Nano/Small char box     640 x 640         same 3.7:1 upscale            /255       640
    #   M3A PicoDet-S/M char box           416 x 416         crop upscaled to 1550x416 (3.7:1) /255    416
    #   M3A cls  Char classifier (MBv2)    64 x 64           square stretch                ImageNet   n/a
    #   M3A OCR  ResNetCRNN (CTC)          256 x 64 (WxH)    SmartResize pad               none       n/a
    #   M3B     Province (gray ResNet)     256 x 80 / 64x256 (WxH) SmartResize pad         ImageNet   n/a
    #
    # M1.5 / M3A-cls / OCR / M3B are torchvision transforms wired in api_server.py
    # __init__ (tf_country / tf_char / tf_ocr / tf_prov) and must match
    # train_country_classifier.py / train_ocr.py / train_*_province.py.
    #
    # OVERRIDES FOR A/B TESTING (optional):
    #   Set "tensor" (square network size) and/or "aspect" (W/H of the pixels
    #   fed to the stretch) per stage. Omitted or None keys keep the trained
    #   default. Everything downstream — serve geometry, imgsz, RF-DETR
    #   predict(shape=...), box rescaling, startup banner — follows automatically.
    #
    #   * Only values the checkpoint actually supports make sense:
    #       - M1 D-FINE:    tensor 640 (trained) — 416/512 also valid at predict
    #                       time for a quick scale-robustness test.
    #       - M1 PicoDet:   tensor 416 (trained) — the ONNX is hard-pinned 416.
    #       - M1/M3A RF-DETR: Base 560, Small 512, Nano 384 (ONNX-pinned).
    #       - M2:           tensor 640 (D-FINE) / 560 (RF-DETR-Base).
    #   * "aspect" only changes the PIXELS you feed before the stretch — the
    #     tensor stays square. e.g. M2 aspect 4/3 tests serving rectified
    #     plates at 4:3 instead of the trained ~2:1.
    #   * For a PERMANENT change: re-train AND re-export ONNX at the new size,
    #     then update the table above.
    #   * Env-var equivalent (no restart/edit): LPR_PRE_M1_TENSOR, LPR_PRE_M1_ASPECT,
    #     LPR_PRE_M2_TENSOR, LPR_PRE_M2_ASPECT, LPR_PRE_M3A_TENSOR, LPR_PRE_M3A_ASPECT.
    #
    # Examples — uncomment to activate:
    # PREPROCESS_OVERRIDES = {
    #     "M1":  {"tensor": 416, "aspect": 1.333},   # serve M1 like PicoDet geometry
    #     "M2":  {"aspect": 4/3},                    # test 4:3 plate crops instead of 2:1
    #     "M3A": {"tensor": 640},                    # serve RF-DETR-Base charbox at 640
    # }
    PREPROCESS_OVERRIDES: dict = {}   # keep {} (= all trained defaults) for production

    # ============================================================
    # (No need to edit below unless you know what you're doing)
    # ============================================================

    # --- System & Paths ---
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    _force_cpu_env = os.environ.get("FORCE_CPU", "0") == "1" or os.environ.get("DEVICE", "").lower() == "cpu"
    # Merge env variable with the FORCE_CPU toggle above (either one activates CPU mode)
    @property
    def DEVICE(self):
        use_cpu = self.FORCE_CPU or self._force_cpu_env
        if use_cpu:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    NUM_WORKERS = 0 
    
    # Data Paths
    DATA_DIR  = PROJECT_ROOT / "data"
    CROPS_DIR = PROJECT_ROOT / "output" / "ground_truth_crops"
    TRAIN_CSV = CROPS_DIR / "train.csv"
    VAL_CSV   = CROPS_DIR / "val.csv"
    TEST_CSV  = CROPS_DIR / "val.csv"

    # --- Model Weights & Architecture Tags (Centralized for Dashboard) ---
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    
    # Model 1: Plate Detection
    MODEL_1_PATH = WEIGHTS_DIR / "plate_polygon_detector.pt"
    MODEL_1_RTDETR_PATH = WEIGHTS_DIR / "plate_detector_rtdetr.pt"
    MODEL_1_RFDETR_PATH = WEIGHTS_DIR / "plate_detector_rfdetr.pt"
    MODEL_1_RFDETR_SMALL_PATH = WEIGHTS_DIR / "plate_detector_rfdetr_small.pt"
    PLATE_CORNER_MODEL_PATH = WEIGHTS_DIR / "plate_corner_regressor.pth"

    @property
    def ACTIVE_MODEL_1_PATH(self):
        # Use MODEL_1_FILENAME from Easy Config first
        chosen = self.WEIGHTS_DIR / self.MODEL_1_FILENAME
        if chosen.exists():
            return chosen
        # Fallback chain
        if self.MODEL_1_RFDETR_SMALL_PATH.exists():
            return self.MODEL_1_RFDETR_SMALL_PATH
        if self.MODEL_1_RFDETR_PATH.exists():
            return self.MODEL_1_RFDETR_PATH
        if self.MODEL_1_RTDETR_PATH.exists():
            return self.MODEL_1_RTDETR_PATH
        return self.MODEL_1_PATH

    @property
    def MODEL_1_TAG(self):
        name = self.ACTIVE_MODEL_1_PATH.name.lower()
        if "picodet_m" in name:
            return "PicoDet-M (PaddleDetection / Apache-2.0, ~12 ms CPU)"
        if "picodet" in name:
            return "PicoDet-S (PaddleDetection / Apache-2.0, ~7 ms CPU)"
        if "dfine_nano" in name:
            return "D-FINE Nano (MIT, ~25 ms CPU)"
        if "dfine_small" in name:
            return "D-FINE Small (MIT, ~45 ms CPU)"
        if "dfine" in name:
            return "D-FINE (MIT)"
        if "rfdetr_obb" in name:
            return "RF-DETR-OBB Small (Apache-2.0 / MIT)"
        if "rfdetr_nano" in name:
            return "RF-DETR-Nano (Apache-2.0, ⚡)"
        if "rfdetr_small" in name:
            return "RF-DETR-Small (Apache-2.0)"
        if "rfdetr" in name:
            return "RF-DETR-Base (Apache-2.0)"
        if "rtdetrv2" in name:
            return "RT-DETRv2-R18 (Apache-2.0)"
        if "rtdetr" in name:
            if self.PLATE_CORNER_MODEL_PATH.exists():
                return "RT-DETR-L + Polygon Quad (Apache-2.0 / BSD-3)"
            return "RT-DETR-L (Apache-2.0)"
        return "YOLO11-seg (Polygon)"
    
    # Model 1.5: Country Classification
    MODEL_1_5_PATH = WEIGHTS_DIR / "country_classifier.pth"
    MODEL_1_5_TAG = "MobileNetV3-Small (Thai vs Laos)"
    
    # Model 2: Component Detection
    MODEL_2_PATH = WEIGHTS_DIR / "component_detector.pt"
    MODEL_2_RTDETR_PATH = WEIGHTS_DIR / "component_detector_rtdetr.pt"
    MODEL_2_RFDETR_PATH = WEIGHTS_DIR / "component_detector_rfdetr.pt"
    MODEL_2_RFDETR_SMALL_PATH = WEIGHTS_DIR / "component_detector_rfdetr_small.pt"
    MODEL_2_RFDETR_NANO_PATH = WEIGHTS_DIR / "component_detector_rfdetr_nano.pt"
    MODEL_2_DFINE_NANO_PATH = WEIGHTS_DIR / "component_detector_dfine_nano.pt"

    @property
    def ACTIVE_MODEL_2_PATH(self):
        chosen = self.WEIGHTS_DIR / self.MODEL_2_FILENAME
        if chosen.exists():
            return chosen
        if self.MODEL_2_DFINE_NANO_PATH.exists():
            return self.MODEL_2_DFINE_NANO_PATH
        if self.MODEL_2_RFDETR_NANO_PATH.exists():
            return self.MODEL_2_RFDETR_NANO_PATH
        if self.MODEL_2_RFDETR_SMALL_PATH.exists():
            return self.MODEL_2_RFDETR_SMALL_PATH
        if self.MODEL_2_RFDETR_PATH.exists():
            return self.MODEL_2_RFDETR_PATH
        if self.MODEL_2_RTDETR_PATH.exists():
            return self.MODEL_2_RTDETR_PATH
        return self.MODEL_2_PATH

    @property
    def MODEL_2_TAG(self):
        name = self.ACTIVE_MODEL_2_PATH.name.lower()
        if "picodet_m" in name:
            return "PicoDet-M (PaddleDetection / Apache-2.0, ⚡)"
        if "picodet" in name:
            return "PicoDet-S (PaddleDetection / Apache-2.0, ⚡)"
        if "dfine_small" in name:
            return "D-FINE Small (MIT, ⚡)"
        if "dfine_nano" in name:
            return "D-FINE Nano (MIT, ⚡)"
        if "rfdetr_nano" in name:
            return "RF-DETR-Nano (Apache-2.0, ⚡)"
        if "rfdetr_small" in name:
            return "RF-DETR-Small (Apache-2.0)"
        if "rfdetr" in name:
            return "RF-DETR-Base (Apache-2.0)"
        if "rtdetr" in name:
            return "RT-DETR-L (Apache-2.0)"
        return "YOLO11-Comp (plate_char / prov)"
    
    # Model 3A: Character Box Detection & Recognition
    CHAR_BOX_MODEL_PATH = WEIGHTS_DIR / "character_box_detector.pt"
    CHAR_BOX_MODEL_RTDETR_PATH = WEIGHTS_DIR / "character_box_detector_rtdetr.pt"
    CHAR_BOX_MODEL_RFDETR_PATH = WEIGHTS_DIR / "character_box_detector_rfdetr.pt"
    CHAR_BOX_MODEL_RFDETR_SMALL_PATH = WEIGHTS_DIR / "character_box_detector_rfdetr_small.pt"
    CHAR_BOX_MODEL_RFDETR_NANO_PATH = WEIGHTS_DIR / "character_box_detector_rfdetr_nano.pt"
    CHAR_BOX_MODEL_DFINE_NANO_PATH = WEIGHTS_DIR / "character_box_detector_dfine_nano.pt"
    CHAR_BOX_MODEL_DFINE_SMALL_PATH = WEIGHTS_DIR / "character_box_detector_dfine_small.pt"

    @property
    def ACTIVE_CHAR_BOX_MODEL_PATH(self):
        chosen = self.WEIGHTS_DIR / self.MODEL_3A_FILENAME
        if chosen.exists():
            return chosen
        if self.CHAR_BOX_MODEL_DFINE_SMALL_PATH.exists():
            return self.CHAR_BOX_MODEL_DFINE_SMALL_PATH
        if self.CHAR_BOX_MODEL_DFINE_NANO_PATH.exists():
            return self.CHAR_BOX_MODEL_DFINE_NANO_PATH
        if self.CHAR_BOX_MODEL_RFDETR_NANO_PATH.exists():
            return self.CHAR_BOX_MODEL_RFDETR_NANO_PATH
        if self.CHAR_BOX_MODEL_RFDETR_SMALL_PATH.exists():
            return self.CHAR_BOX_MODEL_RFDETR_SMALL_PATH
        if self.CHAR_BOX_MODEL_RFDETR_PATH.exists():
            return self.CHAR_BOX_MODEL_RFDETR_PATH
        if self.CHAR_BOX_MODEL_RTDETR_PATH.exists():
            return self.CHAR_BOX_MODEL_RTDETR_PATH
        return self.CHAR_BOX_MODEL_PATH

    @property
    def CHAR_BOX_TAG(self):
        name = self.ACTIVE_CHAR_BOX_MODEL_PATH.name.lower()
        if "picodet_m" in name:
            return "PicoDet-M (PaddleDetection / Apache-2.0, ⚡)"
        if "picodet" in name:
            return "PicoDet-S (PaddleDetection / Apache-2.0, ⚡)"
        if "dfine_small" in name:
            return "D-FINE Small (MIT, ⚡)"
        if "dfine_nano" in name:
            return "D-FINE Nano (MIT, ⚡)"
        if "rfdetr_nano" in name:
            return "RF-DETR-Nano (Apache-2.0, ⚡)"
        if "rfdetr_small" in name:
            return "RF-DETR-Small (Apache-2.0)"
        if "rfdetr" in name:
            return "RF-DETR-Base (Apache-2.0)"
        if "rtdetr" in name:
            return "RT-DETR-L (Apache-2.0)"
        return "YOLO11-Box"
    
    @property
    def CHAR_CLASS_THAI_PATH(self):
        return self.WEIGHTS_DIR / self.CHAR_CLASSIFIER_THAI_FILENAME
    
    CHAR_CLASS_THAI_TAG = "MobileNetV2 (50 Thai Classes)"
    
    CHAR_CLASS_LAO_PATH = WEIGHTS_DIR / "character_classifier_lao.pth"
    CHAR_CLASS_LAO_TAG = "MobileNetV2 (34 Lao Classes)"
    
    OCR_MODEL_PATH = WEIGHTS_DIR / "ocr_model.pth"
    OCR_MODEL_TAG = "ResNetCRNN (CTC)"
    
    # Model 3B: Province Classification
    PROV_MODEL_THAI_PATH = WEIGHTS_DIR / "province_model.pth"
    PROV_MODEL_THAI_GRAYSCALE_PATH = WEIGHTS_DIR / "province_model_grayscale_thai.pth"
    PROV_MODEL_THAI_RESNET34_PATH = WEIGHTS_DIR / "province_model_resnet34_grayscale_thai.pth"

    PROV_MODEL_LAO_PATH = WEIGHTS_DIR / "province_model_lao.pth"
    PROV_MODEL_LAO_GRAYSCALE_PATH = WEIGHTS_DIR / "province_model_grayscale_lao.pth"

    @property
    def ACTIVE_PROV_MODEL_THAI_PATH(self):
        chosen = self.WEIGHTS_DIR / self.MODEL_3B_THAI_FILENAME
        if chosen.exists():
            return chosen
        if self.PROV_MODEL_THAI_RESNET34_PATH.exists():
            return self.PROV_MODEL_THAI_RESNET34_PATH
        if self.PROV_MODEL_THAI_GRAYSCALE_PATH.exists():
            return self.PROV_MODEL_THAI_GRAYSCALE_PATH
        return self.PROV_MODEL_THAI_PATH

    @property
    def PROV_MODEL_THAI_TAG(self):
        name = self.ACTIVE_PROV_MODEL_THAI_PATH.name.lower()
        if "resnet34" in name:
            return "ResNet34-Grayscale (77 Thai Provinces, 256x80)"
        if "grayscale" in name:
            return "ResNet18-Grayscale (77 Thai Provinces)"
        if self.PROV_MODEL_THAI_PATH.exists():
            return "ResNet34 (77 Thai Provinces)"
        return "MobileNetV2 (77 Thai Provinces)"

    @property
    def ACTIVE_PROV_MODEL_LAO_PATH(self):
        chosen = self.WEIGHTS_DIR / self.MODEL_3B_LAO_FILENAME
        if chosen.exists():
            return chosen
        if self.PROV_MODEL_LAO_GRAYSCALE_PATH.exists():
            return self.PROV_MODEL_LAO_GRAYSCALE_PATH
        return self.PROV_MODEL_LAO_PATH

    @property
    def PROV_MODEL_LAO_TAG(self):
        if self.PROV_MODEL_LAO_GRAYSCALE_PATH.exists():
            return "ResNet18-Grayscale (18 Lao Provinces)"
        if self.PROV_MODEL_LAO_PATH.exists():
            return "ResNet18 (18 Lao Provinces)"
        return "MobileNetV2 (18 Lao Provinces)"

    # Aliases for backwards compatibility
    MODEL_DETECTION_PATH = MODEL_1_PATH
    MODEL_OCR_PREP_PATH  = MODEL_2_PATH
    OCR_MODEL_SAVE_PATH  = OCR_MODEL_PATH
    PROV_MODEL_SAVE_PATH = PROV_MODEL_THAI_PATH

    @property
    def model_tags(self) -> dict:
        return {
            "model_1": self.MODEL_1_TAG,
            "model_1_5": self.MODEL_1_5_TAG,
            "model_2": self.MODEL_2_TAG,
            "char_box": self.CHAR_BOX_TAG,
            "char_class_thai": self.CHAR_CLASS_THAI_TAG,
            "char_class_lao": self.CHAR_CLASS_LAO_TAG,
            "ocr_ctc": self.OCR_MODEL_TAG,
            "prov_thai": self.PROV_MODEL_THAI_TAG,
            "prov_lao": self.PROV_MODEL_LAO_TAG,
        }

    # Misc
    CHAR_MAP_PATH = WEIGHTS_DIR / "int_to_char.json"
    PROV_MAP_PATH = WEIGHTS_DIR / "province_map.json"
    PROV_ABBR_MAP_PATH = WEIGHTS_DIR / "province_abbr_map.json"
    THAI_PROVINCE_CROPS_DIR = PROJECT_ROOT / "datasets" / "thai_province_crops"
    THAI_CHAR_CROPS_DIR = PROJECT_ROOT / "datasets" / "thai_character_crops"
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # --- Hyperparameters ---
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE_OCR = 32
    BATCH_SIZE_PROV = 32

    # --- Transformations ---
    OCR_TARGET_SIZE = (64, 256)
    PROV_TARGET_SIZE = (224, 224)
    
    AUG_DEGREES = 15
    AUG_TRANSLATE = (0.05, 0.05) 
    AUG_SCALE = (0.8, 1.2)
    AUG_SHEAR = 10 
    AUG_PERSPECTIVE = 0.5 
    AUG_COLOR_JITTER = (0.5, 0.5, 0.5, 0.1)
    AUG_BLUR_SIGMA = (0.1, 1.5)

    # --- Debugging ---
    DEBUG_MODE = True
    DEBUG_IMAGE_DIR = PROJECT_ROOT / "debug_api"

    # --- Authentication & User Database ---
    AUTH_ENABLED = True
    ALLOW_DEV_ADMIN = True
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "thai_lpr_jwt_super_secret_key_2026")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRES_DAYS = 7
    USERS_DB_PATH = DATA_DIR / "lpr_users.db"
    FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    FIRESTORE_USERS_COLLECTION = "users"

cfg = Config()
cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
cfg.DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)