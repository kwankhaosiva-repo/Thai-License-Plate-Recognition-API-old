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
    # Options (fastest → most accurate):
    #   "plate_detector_dfine_nano.pt"     ← D-FINE-Nano (recommended ⚡, 25 ms CPU)
    #   "plate_detector_dfine_small.pt"    ← D-FINE-Small (more accurate)
    #   "plate_detector_rfdetr_small.pt"   ← RF-DETR-Small
    #   "plate_detector_rfdetr.pt"         ← RF-DETR-Base  (more accurate)
    #   "plate_detector_rtdetr.pt"         ← RT-DETR-L     (older)
    MODEL_1_FILENAME = "plate_detector_dfine_nano.pt"

    # --- Model 2: Component Detector (plate_char / province bbox) ---
    #   "component_detector_rfdetr_small.pt"   ← RF-DETR-Small (recommended ⚡)
    #   "component_detector_rfdetr.pt"         ← RF-DETR-Base
    #   "component_detector_rtdetr.pt"         ← RT-DETR-L (older)
    MODEL_2_FILENAME = "component_detector_rfdetr_small.pt"

    # --- Model 3A: Character Box Detector ---
    #   "character_box_detector_rfdetr_small.pt"  ← RF-DETR-Small (recommended ⚡)
    #   "character_box_detector_rfdetr.pt"        ← RF-DETR-Base
    #   "character_box_detector_rtdetr.pt"        ← RT-DETR-L (older)
    MODEL_3A_FILENAME = "character_box_detector_rfdetr_small.pt"

    # --- Model 3A: OCR (CTC Text Recognition) ---
    OCR_FILENAME = "ocr_model.pth"

    # --- Model 3B: Thai Province Classifier ---
    #   "province_model_grayscale_thai.pth"  ← Grayscale ResNet18 (recommended ⚡)
    #   "province_model.pth"                 ← ResNet34
    MODEL_3B_THAI_FILENAME = "province_model_grayscale_thai.pth"

    # --- Lao Plate Detector ---
    #   "plate_detector_lao_rfdetr_small.pt"  ← RF-DETR-Small (recommended ⚡)
    #   "plate_detector_lao_rfdetr.pt"        ← RF-DETR-Base
    MODEL_LAO_FILENAME = "plate_detector_lao_rfdetr_small.pt"

    # --- Model 3B: Lao Province Classifier ---
    MODEL_3B_LAO_FILENAME = "province_model_grayscale_lao.pth"

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
        if "dfine_nano" in name:
            return "D-FINE Nano (MIT, ~25 ms CPU)"
        if "dfine_small" in name:
            return "D-FINE Small (MIT, ~45 ms CPU)"
        if "dfine" in name:
            return "D-FINE (MIT)"
        if "rfdetr_small" in name:
            return "RF-DETR-Small (Apache-2.0)"
        if "rfdetr" in name:
            return "RF-DETR-Base (Apache-2.0)"
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

    @property
    def ACTIVE_MODEL_2_PATH(self):
        chosen = self.WEIGHTS_DIR / self.MODEL_2_FILENAME
        if chosen.exists():
            return chosen
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

    @property
    def ACTIVE_CHAR_BOX_MODEL_PATH(self):
        chosen = self.WEIGHTS_DIR / self.MODEL_3A_FILENAME
        if chosen.exists():
            return chosen
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
        if "rfdetr_small" in name:
            return "RF-DETR-Small (Apache-2.0)"
        if "rfdetr" in name:
            return "RF-DETR-Base (Apache-2.0)"
        if "rtdetr" in name:
            return "RT-DETR-L (Apache-2.0)"
        return "YOLO11-Box"
    
    CHAR_CLASS_THAI_PATH = WEIGHTS_DIR / "character_classifier.pth"
    CHAR_CLASS_THAI_TAG = "MobileNetV2 (50 Thai Classes)"
    
    CHAR_CLASS_LAO_PATH = WEIGHTS_DIR / "character_classifier_lao.pth"
    CHAR_CLASS_LAO_TAG = "MobileNetV2 (34 Lao Classes)"
    
    OCR_MODEL_PATH = WEIGHTS_DIR / "ocr_model.pth"
    OCR_MODEL_TAG = "ResNetCRNN (CTC)"
    
    # Model 3B: Province Classification
    PROV_MODEL_THAI_PATH = WEIGHTS_DIR / "province_model.pth"
    PROV_MODEL_THAI_GRAYSCALE_PATH = WEIGHTS_DIR / "province_model_grayscale_thai.pth"

    PROV_MODEL_LAO_PATH = WEIGHTS_DIR / "province_model_lao.pth"
    PROV_MODEL_LAO_GRAYSCALE_PATH = WEIGHTS_DIR / "province_model_grayscale_lao.pth"

    @property
    def ACTIVE_PROV_MODEL_THAI_PATH(self):
        chosen = self.WEIGHTS_DIR / self.MODEL_3B_THAI_FILENAME
        if chosen.exists():
            return chosen
        if self.PROV_MODEL_THAI_GRAYSCALE_PATH.exists():
            return self.PROV_MODEL_THAI_GRAYSCALE_PATH
        return self.PROV_MODEL_THAI_PATH

    @property
    def PROV_MODEL_THAI_TAG(self):
        if self.PROV_MODEL_THAI_GRAYSCALE_PATH.exists():
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

cfg = Config()
cfg.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
cfg.DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)