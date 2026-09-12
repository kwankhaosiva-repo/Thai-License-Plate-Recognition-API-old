"""
src/verify_lao_character_crops.py

Generates visual verification montages and an interactive HTML report
for inspecting harvested Lao character crops in datasets/Lao/lao_character_crops/.

Outputs:
  - Montages saved to: output/lao_character_verification/montages/{class_name}.jpg
  - HTML report: output/lao_character_verification/report.html
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CROPS_DIR = PROJECT_ROOT / "datasets" / "Lao" / "lao_character_crops"
TRAIN_DIR = CROPS_DIR / "train"
VALID_DIR = CROPS_DIR / "valid"
OUTPUT_VERIFY_DIR = PROJECT_ROOT / "output" / "lao_character_verification"
MONTAGES_DIR = OUTPUT_VERIFY_DIR / "montages"
MAP_PATH = PROJECT_ROOT / "weights" / "char_classifier_map_lao.json"


def create_character_montage(image_paths: list, thumb_size: int = 64, max_samples: int = 24) -> np.ndarray:
    """Generates an orderly visual contact sheet montage of square character crops."""
    if not image_paths:
        return None

    if len(image_paths) > max_samples:
        indices = np.linspace(0, len(image_paths) - 1, max_samples, dtype=int)
        selected = [image_paths[i] for i in indices]
    else:
        selected = image_paths

    n = len(selected)
    cols = min(6, n)
    rows = int(np.ceil(n / max(cols, 1)))

    pad = 4
    canvas_h = rows * thumb_size + (rows + 1) * pad
    canvas_w = cols * thumb_size + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    for i, p in enumerate(selected):
        r = i // cols
        c = i % cols
        img = cv2.imread(str(p))
        if img is None:
            continue
        resized = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        y = pad + r * (thumb_size + pad)
        x = pad + c * (thumb_size + pad)
        canvas[y:y + thumb_size, x:x + thumb_size] = resized

    return canvas


def generate_character_verification_report():
    print("=======================================================")
    print("--- Generating Lao Character Visual Verification Report ---")
    print(f"Crops Directory: {CROPS_DIR}")
    print(f"Output Directory: {OUTPUT_VERIFY_DIR}")
    print("=======================================================")

    MONTAGES_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train directory not found: {TRAIN_DIR}")

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_char = json.load(f)

    classes = [idx_to_char[str(i)] for i in range(len(idx_to_char))]

    report_items = []
    total_train = 0
    total_valid = 0

    for cls_name in classes:
        t_dir = TRAIN_DIR / cls_name
        v_dir = VALID_DIR / cls_name

        t_files = list(t_dir.glob("*.jpg")) + list(t_dir.glob("*.png")) if t_dir.exists() else []
        v_files = list(v_dir.glob("*.jpg")) + list(v_dir.glob("*.png")) if v_dir.exists() else []

        n_train = len(t_files)
        n_valid = len(v_files)
        total_train += n_train
        total_valid += n_valid

        # Create montage of sample crops (combine train and valid)
        sample_pool = t_files + v_files
        montage = create_character_montage(sample_pool, thumb_size=64, max_samples=24)
        montage_fn = f"char_{cls_name}.jpg"
        montage_path = MONTAGES_DIR / montage_fn

        if montage is not None:
            cv2.imwrite(str(montage_path), montage)

        category = "Digit" if cls_name.isdigit() else "Lao Consonant"
        report_items.append({
            "class": cls_name,
            "category": category,
            "train_count": n_train,
            "valid_count": n_valid,
            "total_count": n_train + n_valid,
            "montage_rel": f"montages/{montage_fn}",
            "status": "PASS" if (n_train >= 1 and n_valid >= 1) else "ATTENTION"
        })

    # Build sleek HTML Report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Lao Character Crops - Visual Verification Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --success: #22c55e;
      --warn: #eab308;
    }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .header {{
      max-width: 1200px;
      margin: 0 auto 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 0; font-size: 24px; color: var(--accent); }}
    .stats-bar {{
      display: flex;
      gap: 16px;
    }}
    .stat-pill {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 16px;
      text-align: center;
    }}
    .stat-pill .num {{ font-size: 20px; font-weight: bold; color: var(--text); }}
    .stat-pill .lbl {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); }}
    .grid {{
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .card-header {{
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
    }}
    .char-badge {{
      font-size: 28px;
      font-weight: bold;
      color: #38bdf8;
      background: rgba(56, 189, 248, 0.1);
      width: 48px;
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
    }}
    .char-meta {{
      text-align: right;
    }}
    .badge-pass {{
      background: rgba(34, 197, 94, 0.15);
      color: var(--success);
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: bold;
    }}
    .card-body {{
      padding: 12px;
      display: flex;
      justify-content: center;
      align-items: center;
      background: #111827;
    }}
    .card-body img {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }}
    .card-footer {{
      padding: 8px 16px;
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: var(--text-muted);
      border-top: 1px solid var(--border);
    }}
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🇱🇦 Lao Character Crops - Visual Verification Report</h1>
      <p style="margin: 4px 0 0; color: var(--text-muted); font-size: 13px;">
        Harvested with RT-DETR Character Box Detector + Ground Truth Sequence (ground_truth_all.csv)
      </p>
    </div>
    <div class="stats-bar">
      <div class="stat-pill">
        <div class="num">{len(classes)}</div>
        <div class="lbl">Classes</div>
      </div>
      <div class="stat-pill">
        <div class="num">{total_train:,}</div>
        <div class="lbl">Train Crops</div>
      </div>
      <div class="stat-pill">
        <div class="num">{total_valid:,}</div>
        <div class="lbl">Valid Crops</div>
      </div>
      <div class="stat-pill">
        <div class="num">{total_train + total_valid:,}</div>
        <div class="lbl">Total Crops</div>
      </div>
    </div>
  </div>

  <div class="grid">
"""

    for item in report_items:
        html_content += f"""
    <div class="card">
      <div class="card-header">
        <div class="char-badge">{item['class']}</div>
        <div class="char-meta">
          <span class="badge-pass">{item['status']}</span>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">{item['category']}</div>
        </div>
      </div>
      <div class="card-body">
        <img src="{item['montage_rel']}" alt="Class {item['class']}">
      </div>
      <div class="card-footer">
        <span>Train: <b>{item['train_count']}</b></span>
        <span>Valid: <b>{item['valid_count']}</b></span>
        <span>Total: <b>{item['total_count']}</b></span>
      </div>
    </div>
"""

    html_content += """
  </div>
</body>
</html>
"""

    report_path = OUTPUT_VERIFY_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Verification Report successfully generated at:")
    print(f"  {report_path}")
    print("=======================================================\n")
    return report_path


if __name__ == "__main__":
    generate_character_verification_report()
