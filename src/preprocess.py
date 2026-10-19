import io
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms

class SmartResize:
    def __init__(self, size, mode="L"):
        self.size = size
        self.mode = mode

    def __call__(self, img):
        if not hasattr(img, 'size'):
            img = Image.fromarray(img)
        
        if img.mode != self.mode:
            img = img.convert(self.mode)

        w, h = img.size
        target_w, target_h = self.size
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        
        new_img = Image.new(self.mode, (target_w, target_h), 0)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        
        return new_img

def preprocess_raw_api_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def _ocr_enhanced_gray(img):
    """Train/serve contract (v2 audit fix #1): PIL.ImageOps.autocontrast(cutoff=1)
    on the grayscale crop BEFORE SmartResize. Must stay identical to the serve
    path in api_server.py (char_crop -> convert('L') -> tf_ocr); the op used to
    exist ONLY at serve time, which was a real train/serve mismatch."""
    if img.mode != "L":
        img = img.convert("L")
    return ImageOps.autocontrast(img, cutoff=1)


def get_ocr_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            _ocr_enhanced_gray,
            SmartResize((256, 64), mode="L"),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5)
            ], p=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.ToTensor(), # 0-1
        ])
    else:
        return transforms.Compose([
            _ocr_enhanced_gray,
            SmartResize((256, 64), mode="L"),
            transforms.ToTensor(),
        ])

def get_prov_transforms(is_train=False):
    if is_train:
        return transforms.Compose([
            SmartResize((256, 64), mode="RGB"),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomAffine(degrees=4, translate=(0.02, 0.04)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.4)
            ], p=0.6),
            transforms.RandomAutocontrast(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            SmartResize((256, 64), mode="RGB"),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])


class GrayscaleSmartResize:
    """
    Color-invariant resize that dynamically pads with natural edge/corner tones
    instead of harsh pitch-black zero padding.
    Supports toggling ambient padding and grayscale normalization for pure model benchmarking.
    """
    def __init__(self, size=(256, 64), use_ambient_padding=None, use_grayscale=None):
        self.target_w, self.target_h = size
        self._use_ambient_padding = use_ambient_padding
        self._use_grayscale = use_grayscale

    @property
    def use_ambient_padding(self):
        if self._use_ambient_padding is not None:
            return self._use_ambient_padding
        try:
            from src.config import Config
            return Config.is_ambient_padding_enabled()
        except Exception:
            return True

    @property
    def use_grayscale(self):
        if self._use_grayscale is not None:
            return self._use_grayscale
        try:
            from src.config import Config
            return Config.is_grayscale_norm_enabled()
        except Exception:
            return True

    def __call__(self, img_pil):
        if not hasattr(img_pil, "size"):
            img_pil = Image.fromarray(img_pil)

        # 1. Grayscale vs Color
        if self.use_grayscale:
            base_im = img_pil.convert("L")
        else:
            base_im = img_pil.convert("RGB")

        w, h = base_im.size

        # 2. Ambient corner padding vs Pure direct stretch
        if not self.use_ambient_padding:
            resized = base_im.resize((self.target_w, self.target_h), Image.BICUBIC)
            return resized.convert("RGB")

        im_arr = np.array(base_im)
        if self.use_grayscale:
            corners = [im_arr[0, 0], im_arr[0, -1], im_arr[-1, 0], im_arr[-1, -1]]
            bg_val = int(np.median(corners))
            new_im = Image.new("L", (self.target_w, self.target_h), color=bg_val)
        else:
            corners = [im_arr[0, 0], im_arr[0, -1], im_arr[-1, 0], im_arr[-1, -1]]
            bg_val = tuple(np.median(corners, axis=0).astype(np.uint8))
            new_im = Image.new("RGB", (self.target_w, self.target_h), color=bg_val)

        ratio = min(self.target_w / w, self.target_h / h)
        new_w, new_h = max(4, int(w * ratio)), max(4, int(h * ratio))
        resized = base_im.resize((new_w, new_h), Image.BICUBIC)

        paste_x = (self.target_w - new_w) // 2
        paste_y = (self.target_h - new_h) // 2
        new_im.paste(resized, (paste_x, paste_y))

        return new_im.convert("RGB")


def get_grayscale_prov_transforms(is_train=False, size=(256, 64), use_ambient_padding=None, use_grayscale=None):
    """Grayscale SmartResize transforms for province classifiers.

    RESOLUTION CONTRACT (must match the training scripts exactly):
      - Thai  (train_grayscale_province_thai.py): (256, 80) — backbone resnet18/34
      - Lao   (train_grayscale_province_lao.py) : (256, 64) — backbone resnet18
    Serving at the wrong H silently degrades accuracy (train/serve mismatch).
    """
    resize_op = GrayscaleSmartResize(size, use_ambient_padding=use_ambient_padding, use_grayscale=use_grayscale)
    if is_train:
        return transforms.Compose([
            resize_op,
            transforms.RandomAffine(degrees=4, translate=(0.02, 0.04)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAutocontrast(p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])
    else:
        return transforms.Compose([
            resize_op,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
        ])

