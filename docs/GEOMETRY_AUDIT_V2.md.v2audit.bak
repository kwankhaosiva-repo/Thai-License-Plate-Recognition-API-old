# Geometry & Leakage Audit — v2 Recognition Retrain

> วัดจริงจากไฟล์บนดิสก์ (ไม่ใช่เดา): ดู `scratch/audit_dataset_geometry.py`,
> `scratch/audit_split_leakage.py`, `scratch/audit_phash_leakage.py`,
> `scratch/audit_pad_math.py` — ทุกอย่าง read-only.

## 1. ตารางสัญญาทาง geometry ต่อ model (ที่ user ถาม)

| Stage | Pretrain origin | Pretrain res | Preprocess (pretrain) | Train res (measured) | Train aspect (measured) | Pad? | Serve geometry | Serve pad | Post-process |
|---|---|---|---|---|---|---|---|---|---|
| **M1 PicoDet-S/M** | COCO via LibreYOLO/PaddleDetection | 416×416 | **stretch, no pad** | full scenes | 1.332 (4:3) | ❌ ไม่ pad | 554×416 → stretch 416 | ไม่ | none |
| **M1 D-FINE n/s** | COCO via LibreYOLO | 640×640 | stretch, no pad | full scenes | 1.332 | ❌ | 4:3 → stretch 640 | ไม่ | none |
| **M2** (components) | COCO (ตาม variant) | 416/640 stretch | stretch, no pad | rectified plates **162×84** | **1.941** | ❌ | 320×160 → **upscale 1280×640** → stretch 640/416 | ไม่ | divide by 4.0 |
| **M3A box** (charbox) | COCO (RF-DETR/D-FINE/PicoDet) | 560/640/416 stretch | stretch, no pad | char rows **215×58** | **3.712** | ❌ | char row → **2386×640 (DFINE)** / **2088×560 (RFDETR)** → stretch | ไม่ | divide per-axis |
| **M3A cls** (MobileNetV2) | ImageNet-1k | 224×224 | ImageNet norm | **64×64 square pre-padded** (3384/3384 = 64×64) | 1.0 | ✅ **corner-median pad ที่ harvest time** | corner-median pad → Resize(64) | ✅ เหมือนกัน | argmax |
| **M3A OCR** (ResNet18+BiLSTM+CTC) | ImageNet-1k | 224×224 | ImageNet norm | 256×64 gray | 4.0 | ✅ **letterbox pad=0 (ดำ)** | SmartResize((256,64),'L') | ✅ เหมือนกัน | CTC |
| **M3B province TH** (ResNet18) | ImageNet-1k | 224×224 | ImageNet norm | crops med aspect **3.31** (p10 1.77 / p90 5.80) | — | ✅ **GrayscaleSmartResize(256,80) letterbox + corner-median** | เหมือนกัน (256,80), norm 0.5/0.25 | ✅ | softmax |
| **M3B province LA** (ResNet18) | ImageNet-1k | 224×224 | ImageNet norm | **3626/4000 เป็น 256×64 อยู่แล้ว** | 4.0 | ✅ GrayscaleSmartResize(256,64) | เหมือนกัน | ✅ | softmax |

## 2. คำตอบเฉพาะคำถาม "padding ที่เอาเข้า"

| ถาม | คำตอบที่วัดจริง |
|---|---|
| **Padding ที่เอาเข้า M1/M2/M3A-box** | **ไม่มี** — ทุกตัว stretch ลง square tensor ตรงๆ (ไม่ letterbox, ไม่ pad) |
| **Padding pretrain** | ไม่มี pad — RF-DETR/D-FINE/PicoDet fine-tune ด้วย pure stretch |
| **Padding province (train & serve)** | letterbox ด้วย **โทนพื้นหลัง (median 4 มุม)** ไม่ใช่ดำ — `GrayscaleSmartResize` ตรงกันทั้งสองฝั่ง ✅ |
| **Padding char (train & serve)** | **corner-median square pad → 64×64** ตรงกันทั้งสองฝั่ง ✅ |
| **Padding OCR** | letterbox **สีดำ (0)** ตรงกันทั้งสองฝั่ง ✅ |

## 3. Mismatch ที่เจอจริง (ต้องแก้ — ไม่ใช่เรื่อง pad)

1. **`ImageOps.autocontrast` มีแค่ฝั่ง serve** (`api_server.py:2033` char cls cutoff=2, `:2154/:2443` OCR cutoff=1) **แต่ไม่มีใน train script/dataset เลย** → train/serve mismatch จริง (ประเภทเดียวกับ CLAHE ที่ถูกลบออกจาก M3A charbox ไปแล้ว)
2. **M2 "upscale 1280×640" เป็น no-op ทางเรขาคณิตสำหรับ plate 2:1** (`scratch/audit_pad_math.py` พิสูจน์: mean|diff| = 0.5/255) — comment ใน `api_server.py:1752-1755` เข้าใจผิดว่า object เล็กลง 3.5×; จริงๆ มีผลเฉพาะเมื่อ crop aspect ≠ train aspect
3. **Docstring `train_grayscale_province_thai.py` เขียนผิด** ว่าใช้ (256,64) แต่ code ใช้ (256,80); `config.py:62` comment ว่า ResNet18 file เป็น "legacy 256x64" ซึ่งไม่จริง (ไฟล์ที่ serve ถูก train ที่ 256,80)
4. **`province_model_resnet34_grayscale_thai.pth` (85 MB) ถูก train แล้วแต่ไม่ได้ serve** (`MODEL_3B_THAI_FILENAME` ชี้ ResNet18)

## 4. Data leakage ที่วัดจริง (เหตุผลที่ accuracy เก่าฟอก)

| Dataset | หลักฐาน |
|---|---|
| **M3A charbox** | **8 timestamp** เดียวกัน (เฟรมวินาทีเดียวกัน = รถคันเดียวกัน) อยู่ทั้ง train และ valid |
| **M3B province** | **20 crops** ชื่อไฟล์เดียวกันเป๊ะอยู่ทั้ง train และ valid — เพราะ `extract_thai_province_crops.py` copy seed จาก valid/test ไป train แล้ว aug; train 9120 รูป = 5935 `aug_real_` + 412 `aug_truck_` + 2345 `rf_` + 428 `gt_` (≈70% เป็น synthetic) |
| **M3A char cls** | `merge_candidates_to_dataset.py` ใส่ 85/15 **ระดับ crop** ด้วย `np.random.rand()` → crop จาก plate เดียวกันหลุดทั้งสองฝั่ง |

## 5. แหล่ง GT ที่ยังอยู่ (ใช้ retrain ได้)

| Label | แหล่ง | สถานะ |
|---|---|---|
| Province TH (77) | `thai-car-license-plate-province.v5i.yolov11` 3219 plate crops 192×99, 1 YOLO line = abbr class → `province_abbr_map.json` | ✅ ครบ, split เป็น Roboflow |
| Char TH (50) | `thai_character_crops/metadata.csv` (3384 rows, 545 plates) + `LPR 2 - Character Box Detection.yolov11` (658 char-row crops + per-char YOLO boxes) | ✅ ครบ (ไฟล์ `gt_plate_char.csv` โดนลบ แต่ mapping เหลือใน metadata.csv) |
| Province LA (18) | `lao_province_crops` + `balance_and_merge_lao_provinces.py` | ⚠️ ต้อง re-extract ด้วย pipeline ใหม่แยก |

## 6. ขั้นตอนที่ทำแล้ว (v2)

- ✅ `src/extract_recognition_crops_v2.py` — re-extract province/char/scene ผ่าน pipeline ปัจจุบัน (M2 → M3A) พร้อม **GT count filter** (ยอมรับเฉพาะเมื่อจำนวน box = จำนวนตัวอักษร GT)
- ✅ `src/split_recognition_dataset_v2.py` — split 80/20 **ระดับรูปต้นฉบับ** (ไม่ใช่ crop) + leakage self-check
- ✅ `tests/test_split_recognition_dataset_v2.py` — 5 unit tests
"""
