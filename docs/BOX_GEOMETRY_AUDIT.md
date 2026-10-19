# BOX & CROP GEOMETRY AUDIT — "กล่องเป็นสี่เหลี่ยมจัตุรัส" / ต้องเทรนใหม่ด้วยอะไร

> วันที่: 2026-10-02 · เครื่องมือวัด: `scratch/diag_boxes.py`, `scratch/diag_m2_input.py`, `scratch/ab_flag_matrix.py`
> ต่อจาก [`docs/PREPROCESS_FLAG_AUDIT.md`](PREPROCESS_FLAG_AUDIT.md)

---

## 0. สรุปสั้น

1. **"picodet detect เป็นสี่เหลี่ยมจัตุรัส" = bug จริงของโค้ดเรา ไม่ใช่ของโมเดล** — `plate_box` ที่ API ส่งกลับถูกตัวแปรในลูปตัวอักษรทับ จนกลายเป็น "กล่องตัวอักษรตัวสุดท้าย" ในพิกัดของ crop ตัวอักษร (แนวตั้ง aspect ~0.54) → **แก้แล้ว**
2. **crop ตัวอักษรถูกตัดขอบล่างจริง** — margin ใต้กล่องตัวอักษร = 2 px (median) และ **16/30 ภาพไม่มี margin ใต้เลย** ขณะที่ชุดเทรน M3A มี margin 8.4% บน / 5.4% ล่าง → **แก้แล้ว** (2 px → 7.5 px, 0/30 ภาพที่ไม่มี margin)
3. **`upscale_to_train_geometry` ไม่ได้ทำอย่างที่คอมเมนต์อ้าง** — มันเป็น geometric no-op เพราะตามด้วย stretch เป็น tensor จัตุรัส → ยกเลิกกันทางคณิตศาสตร์ (ดูข้อ 4)
4. **ห้ามเปลี่ยน `imgsz` ของ PicoDet ที่ inference** — วัดแล้ว 416 → 640 ทำให้ recall **30.0% → 6.7%** ถ้าจะได้ recall ที่ดีขึ้นต้อง **เทรน + export ใหม่ที่ขนาดใหญ่ขึ้น** ไม่ใช่เปลี่ยน imgsz ตอนรัน
5. **M2/M3A ไม่ได้พัง** — province box IoU 0.79 กับ GT, char box aspect 0.57 vs GT 0.59 ⇒ ปัญหาอยู่ที่ *สัญญาอินพุต* และ *post-processing* ไม่ใช่ตัวโมเดล

---

## 1. Bug: `plate_box` ถูกทับด้วยกล่องตัวอักษร (แก้แล้ว)

`process_image()` ใช้ชื่อ `bx1, by1, bx2, by2` ทั้งเป็น **กล่องป้ายของ M1 (พิกัดภาพจริง)** และเป็น **ตัวแปรชั่วคราวในลูปตัวอักษรของ M3A (พิกัดของ crop ตัวอักษร)**:

| บรรทัด (ก่อนแก้) | โค้ด | ผล |
|---|---|---|
| ~2055 | `for b in box_res.boxes: bx1, by1, bx2, by2 = divide_boxes_by_scale(...)` | ทับกล่องป้ายด้วยกล่องตัวอักษรทุกตัว |
| ~2365 | `bx1, by1, bx2, by2 = char_boxes_detail[i]["box"]` | ทับอีกครั้งในบล็อก fusion |
| ~2679 | เหมือนกันในสาขาลาว | ทับอีก |
| ~2910/2914 | วาด overlay `char_box_coords` / `prov_box_coords` | ทับแม้แต่ตอน `debug=True` |

หลักฐาน (ภาพ `147f5356-...jpg`, scene 192×99, crop ตัวอักษร 298×83):

```
ก่อนแก้: plate_box = [253, 3, 295, 81]   (w,h = 42,78  aspect 0.54)   ← อยู่นอกภาพ
         295 < 298 (= กว้าง crop)  และ 81 < 83 (= สูง crop)  → เป็นกล่องตัวอักษรชัดเจน
หลังแก้: plate_box = [0, 0, 192, 93]     (aspect 2.06, อยู่ในขอบภาพ)
```

ผลกระทบ: `plate_box` คือสัญญาของ API — ตัวประมวลผลสตรีมใช้มัน (`api_server.py:3937 cached_box`) และไคลเอนต์ภายนอกใช้ จึงได้กล่องผิดไปด้วย

**วิธีแก้:** ตรึงกล่องป้ายไว้ในตัวแปรเฉพาะทันทีหลัง rectification

```python
m1_plate_box = (int(bx1), int(by1), int(bx2), int(by2))   # ก่อน M1.5
...
"plate_box": list(m1_plate_box),
```

บวกกับเปลี่ยนชื่อตัวแปรในลูปทั้ง 4 จุด (ไทย/ลาว/fusion/overlay) เป็น `cbx1..` / `abx1..` / `obx1..` ให้ตรงกับเจตนา **ยืนยันแล้วว่า `plate_text` และ `province` ไม่เปลี่ยนเลย** (ทดสอบซ้ำภาพเดิม)

---

## 2. Margin ของ crop ตัวอักษร vs สัญญาชุดเทรน (แก้แล้ว)

ชุดเทรน M3A (`LPR 2 - Character Box Detection`) มีกล่องตัวอักษรกินความสูง **8.4% → 94.6%** ของภาพ ⇒ margin บน ~8%, ล่าง ~5%

ของจริงตอน serve (วัด 30 ภาพ):

| | ก่อนแก้ | หลังแก้ | ชุดเทรน |
|---|---|---|---|
| margin เหนือตัวอักษรตัวบนสุด (median) | 4.0 px | **7.0 px** | ~8% ของความสูง |
| margin ใต้ตัวอักษรตัวล่างสุด (median) | 2.0 px | **7.5 px** | ~5% |
| ภาพที่ margin ล่าง ≤ 2 px | **16/30** | **0/30** | — |

สาเหตุ: `pad_cy = min(6, max(2, int(bh_char * 0.04)))` (เล็กเกินไป) + clamp แข็งที่ `cy2 = min(int(rh*0.68), ...)`
แก้เป็น `pad_cy = min(10, max(3, int(bh_char * 0.10)))` และ clamp `0.76*rh`, `pad_cx = min(10, max(3, int(bw*0.04)))`

⚠️ **ข้อจำกัดที่ต้องพูดตรงๆ:** v5i ไม่มี char-level GT ⇒ ยืนยันได้แค่ว่า margin ตรงกับสัญญาเทรนแล้ว (province ไม่เปลี่ยน 30/30, plate_text เปลี่ยน 6/30 ภาพ) **ยังไม่ได้พิสูจน์ว่าอักษรถูกขึ้นหรือไม่** ต้องมี GT ตัวอักษรถึงจะฟันธงได้

---

## 3. โมเดล "แค่โมเดลไม่พอหรอ" — วัดแล้วตอบได้

| คำถาม | คำตอบจากตัวเลข |
|---|---|
| M2 หาโซนจังหวัดเก่งไหม | IoU กับ GT = **0.798** (median), 38/39 ภาพที่ IoU > 0.5 |
| M2 หาโซนตัวอักษรเก่งไหม | aspect ที่ทายได้ 3.22–3.40 (pixel) vs ชุดเทรน 3.84 — ใกล้เคียง |
| M3A หากล่องตัวอักษรเก่งไหม | aspect median **0.57** vs GT **0.59**, กล่องกินความสูง 0.88 ของ crop |
| **แล้วทำไมต้องมี post-processing** | M3A ยังให้กล่องที่กว้างกว่าสูง **19%** (GT มีแค่ 1%) → ต้องมี `split_wide_char_boxes`/NMS; และ DETR ให้กล่องซ้อน 40–65% ที่ conf ต่ำ (คอมเมนต์ในโค้ด) |
| **OpenCV augmentation ตอนเทรน "มั่วซั่ว" ไหม** | ไม่ใช่ — ชุดเทรนคือ Roboflow export (ดูข้อ 6) augmentation เป็นของ Roboflow; โค้ด `warp_perspective_plate` **ไม่เคยอยู่ใน training script ของ M2/M3A/M3B เลย** (อยู่ใน serve + harvester เท่านั้น) |

⇒ สรุป: **โมเดลใช้ได้ แต่ต้องป้อนอินพุตให้ตรงสัญญาเป๊ะ** สิ่งที่ดูเหมือน "OpenCV จำเป็น" จริงๆ คือ 2 อย่างนี้
1. **pixel contract** — crop/inset/letterbox ให้เหมือนตอนเทรน (ข้อ 2 + ambient pad)
2. **post-processing กล่อง** — ซ่อม artifact ของ detector (กล่องกว้าง/กล่องซ้ำ/dash)

---

## 4. `upscale_to_train_geometry` เป็น no-op ทางเรขาคณิต (ข้อควรระวังในเอกสาร)

```
upscale_to_train_geometry(crop, spec): resize เป็น (tensor_w*aspect, tensor_h)   # เช่น M3A 2386x640
ตัวโมเดล: stretch (2386x640) -> (640x640)

สเกลสุทธิ x = (2386/w) * (640/2386) = 640/w      <- ไม่ขึ้นกับ aspect
สเกลสุทธิ y = (640/h)  * (640/640)  = 640/h
```

⇒ ผลลัพธ์ = `cv2.resize(crop, (640,640))` ตรงๆ **การอัปสเกลก่อนไม่ทำให้ aspect ของวัตถุตรงกับตอนเทรน** อย่างที่คอมเมนต์ใน `recommended_input_shape()` อ้าง มันมีผลแค่คุณภาพ interpolation และคณิตศาสตร์การคืนค่า box

**ถ้าต้องการ normalize aspect จริง** ต้องใช้ tensor **ไม่จัตุรัส** — และ repo มี `ResizeMode.ASPECT_STRETCH` เตรียมไว้แล้วแต่ **ยังไม่มีใครใช้** นี่คือช่องทางที่ถูกต้องสำหรับ M2/M3A

---

## 5. PicoDet ต้องใช้ image size เท่าไร — ทดลองแล้ว

ทดลองบน 60 ภาพ (v5i valid, ยิง M1 ตรงๆ ที่ conf 0.35):

| อินพุตให้ M1 | detect | mean conf |
|---|---|---|
| native `imgsz=416` (ค่าปัจจุบัน) | **18/60 = 30.0%** | 0.437 |
| native `imgsz=640` | **4/60 = 6.7%** | 0.440 |
| native `imgsz=1280` | 16/60 = 26.7% | 0.488 |
| **อัปภาพ 3× (LANCZOS4) + `imgsz=416`** | **20/60 = 33.3%** | 0.466 |
| อัปภาพ 3× + `imgsz=640` | 4/60 = 6.7% | 0.425 |

**ข้อสรุป:**
- PicoDet-S v2 ถูก train/export ที่ **416** ⇒ เปลี่ยน imgsz ตอน inference ทำให้ recall **พัง** (30% → 6.7%) **ห้ามทำ**
- ถ้าต้องการ recall บนภาพความละเอียดต่ำ ให้ **อัปสเกลภาพก่อนเข้าโมเดล** (416 คงเดิม) — ทำแล้วในเซสชันก่อน: กู้ภาพที่ M1 มองไม่เห็น 8/14 ⇒ province e2e +1.0 คะแนน (91.89% → 92.93%)
- ถ้าต้องการดีกว่านี้จริง ต้อง **เทรนใหม่ที่ tensor ใหญ่ขึ้น** (เช่น 640) + **export ONNX ที่ 640 ให้ตรงกัน** + ใส่ **scale-degradation augmentation** (ย่อภาพลง 0.25–0.5× แล้วย่อ/ขยายกลับ) เพราะ data จริง 71% ของ crop อยู่ที่ ~61 px กว้าง

**ยังไม่ได้แก้ในโค้ด:** การอัปสเกลอัตโนมัติเมื่อ M1 ไม่เจอ (ต้อง remap พิกัดทั้ง pipeline และ `preview_bgr`/overlay ให้สอดคล้อง) — เสี่ยงเกินกว่าจะแก้แบบครึ่งๆ กลางๆ ในรอบนี้ แนะนำให้ทำพร้อม test เฉพาะ

---

## 6. ที่มาของ dataset — git ตอบไม่ได้ แต่ Roboflow ตอบได้

`git ls-files datasets | wc -l` = **0** ⇒ dataset ทั้งหมด **ไม่ถูก track ใน git** (gitignore) จึงย้อนจาก commit เก่าไม่ได้ ที่มาและวันที่จริงอยู่ใน README ของแต่ละ dataset:

| Dataset | ใช้เทรน | Export จาก Roboflow | ขนาดภาพจริง | หมายเหตุ |
|---|---|---|---|---|
| `thai-car-license-plate-province.v5i.yolov11` | วัดผล e2e (481 valid) | v5 · 2026-09-05 · export 09-07 | 192×99 (465/481) | ป้าย + label = กล่อง **จังหวัด** เท่านั้น (cy 0.86, h 0.232) |
| `LPR 2 - Charactor Detection.yolov11` | **M2** (plate_char/province) | 2026-09-04 | 170×84 | char cy 0.38 h 0.384 · prov cy 0.70 h **0.185** |
| `LPR 2 - Character Box Detection.yolov11` | **M3A** (ต่อตัวอักษร) | 2026-09-07 | 218×58 | กล่องต่อตัวอักษร aspect 0.59, กินความสูง 8.4–94.6% |

**จุดสำคัญ:** v5i (cy 0.86 / h 0.232) กับชุด M2 (cy 0.70 / h 0.185) ใส่กล่องจังหวัด**คนละกรอบ** ⇒ M2 ต้อง generalize ข้ามกรอบ ซึ่งก็ทำได้ (IoU 0.79) แต่หมายความว่า **กรอบของ serve (มาจาก `warp_perspective_plate` ของเราเอง) เป็นกรอบที่ 3** และไม่ตรงกับทั้งสองชุด

---

## 7. สรุปว่าต้องเทรนใหม่ด้วยอะไร (เรียงตามความคุ้มค่า)

| # | งาน | ทำไม | geometry ที่แนะนำ |
|---|---|---|---|
| 1 | **Re-extract ชุด v2 (province/char) ให้ผ่าน serve pipeline จริง** | ตอนนี้ crop เทรนมาจาก Roboflow/mock ไม่เคยผ่าน `warp_perspective_plate` + กติกา inset ของเรา | ใช้ `extract_recognition_crops_v2.py` เวอร์ชันที่เรียก warp เดียวกับ serve |
| 2 | **M3A: เพิ่มความหลากหลาย aspect ของ char-row** | serve ได้ 3.03–3.73 (median 3.31) แต่เทรน 3.76 | เทรนบน crop aspect 3.0–4.5 (+ margin ตามข้อ 2) หรือเปลี่ยนเป็น tensor **ไม่จัตุรัส** `ASPECT_STRETCH` เช่น 1216×320 |
| 3 | **M1: เทรนใหม่ที่ tensor ใหญ่ขึ้น + scale-degradation aug** | 30% recall บน 192×99, และเปลี่ยน imgsz ตอนรันไม่ได้ | เทรนที่ 640 + export ONNX 640 + aug ย่อภาพ 0.25–0.5× |
| 4 | **M2: ไม่ต้องเทรนใหม่** | geometry ตรงสัญญาและ IoU 0.79 | คงไว้ 416, คง `train_aspect_w_over_h=2.0` |
| 5 | **ตัวอักษร (M3A-cls / M3B)**: augment ให้ตรงกับ domain จริง | class imbalance 54:1, บางคลาสมี source เดียว, blur ไม่มีใน aug เลย | เพิ่ม `GaussianBlur(σ 0.5–1.5, p .3)` + `RandomDownscale` |

---

## 8. วิธีรันซ้ำ

```bash
FORCE_CPU=True DIAG_N=40 python -u scratch/diag_boxes.py        # กล่อง M2/M3A + margin
FORCE_CPU=True M2N=40   python -u scratch/diag_m2_input.py      # M2 บน crop ดิบ vs ภาพที่ warp แล้ว
FORCE_CPU=True AB_N=120 python -u scratch/ab_flag_matrix.py     # A/B ทุก flag
FORCE_CPU=True python src/test_classifier_preprocess.py         # ต้อง 17 PASS
```

---

## 9. PicoDet 416 — "pretrained" ตัวไหนกันแน่ และแก้ยังไง

ต้องแยก 3 ชั้นให้ชัด เพราะคำว่า "416" มาจากคนละชั้น:

| ชั้น | ขนาด | มาจากไหน |
|---|---|---|
| Pretrained checkpoint (`LibrePICODETs.pt` / `LibrePICODETm.pt`) | **ไม่ผูกขนาด** | ดาวน์โหลดอัตโนมัติใน `train_picodet_m_plate.py:ensure_pretrained_weights()` เป็น checkpoint สำเร็จรูปของ LibreYOLO (COCO) — backbone ไม่มีขนาดตายตัว |
| **Fine-tune ของเรา** | **416 (stretch)** | `src/train_libreyolo_task.py:40-41` → `"picodet_s": {..., "imgsz": 416}` และ `train_picodet_m_plate.py:53 imgsz: int = 416` |
| **ONNX ที่ deploy** | **416×416 ตายตัว** | ตรวจ graph จริง: `plate_detector_picodet_s_v2.onnx` input = `[1, 3, 416, 416]`, `component_detector_picodet_s_v2.onnx` = `[1, 3, 416, 416]` (export ด้วย `dynamic=False`) |

⇒ "416" ไม่ใช่ข้อจำกัดของโมเดล แต่เป็น **ข้อจำกัดของ export ที่ dynamic=False** ⇒ เปลี่ยนเฉพาะตอน inference ไม่ได้ (และวัดแล้วพัง: 30.0% → 6.7%)

**ถ้าต้องการให้ PicoDet เก่งขึ้นบนภาพความละเอียดต่ำ — ทำตามลำดับนี้:**

1. **ตอนนี้ (ไม่ต้องเทรน):** อัปสเกล *ภาพ* ด้วย LANCZOS4 ก่อนเข้าโมเดล, คง `imgsz=416` → 30.0% → 33.3% และกู้ภาพที่ M1 มองไม่เห็นได้ 8/14 (province e2e +1.0)
2. **เทรนใหม่ที่ 640:** `--imgsz 640` แล้ว **export ONNX ที่ 640 ด้วย** (`dynamic=False` เดิม) แล้วอัปเดต 3 ที่ให้ตรงกัน: `preprocess_registry.M1_PICODET.tensor_(w|h)`, `MODEL_1_FILENAME`, และตารางใน `config.py` (SERVE / PRE-PROCESS GEOMETRY) — `m1_imgsz_native` จะตาม spec เองอัตโนมัติ
3. **พร้อมกันนั้นใส่ scale-degradation augmentation** (สุ่มย่อภาพ 0.25–0.5× แล้วย่อกลับ) เพราะต้นเหตุคือวัตถุเล็ก (71% ของ crop จริงกว้าง ~61 px) ไม่ใช่แค่ขนาด tensor

> หมายเหตุ: ทำไมอัปสเกล 3× แล้วช่วยได้ ทั้งที่สเกลวัตถุใน tensor เท่าเดิม (ภาพถูกบีบเข้า 416 ทั้งคู่) — เพราะมันเปลี่ยน **resampling path**: ย่อจากภาพเล็กตรงๆ ~2–4× ด้วย INTER_LINEAR เกิด aliasing กว่า "อัปด้วย LANCZOS4 แล้วค่อยย่อ" ⇒ ตัวที่ได้ผลคือคุณภาพการรีแซมเพิล ไม่ใช่สเกล

---

## 10. Augmentation ≠ สัญญา serve (ตอบคำถาม "ต้องเหมือนกันทั้งหมดไหม")

ต้องแยก 2 อย่างให้ขาด:

- **Eval-time transform (สัญญา)** — deterministic, ทุกภาพได้ผลเดียวกัน → **serve ต้องเหมือนเป๊ะ** ได้แก่ resize/tensor size, letterbox vs stretch, สี (grayscale), normalize, crop margin, autocontrast *ถ้า* มันอยู่ในพาธ deterministic
- **Train augmentation** — สุ่ม, บางภาพได้บางภาพไม่ได้ → มีไว้สร้าง **invariance** **ไม่ใช่** สัญญา serve

รายโมเดล (อ่านจากโค้ดจริง):

| โมเดล | autocontrast ตอน **eval/เทรน (deterministic)** | augmentation เรื่องคอนทราสต์ | serve ต้องทำตามไหม |
|---|---|---|---|
| Char classifier | **มี** — `train_character_classifier.py:64` รัน `ImageOps.autocontrast(cutoff=2)` ใน `__getitem__` **ทุกภาพ ก่อน augment** | `ColorJitter`, `RandomGrayscale` | **ต้องมี** (cutoff=2) — ไม่งั้น serve ต่างจากข้อมูลเทรนทุกภาพ |
| OCR (CTC) | **มี** — `src/preprocess.py:40 _ocr_enhanced_gray` cutoff=1 ใช้ทั้ง train และ serve | `ColorJitter(b/c 0.5, p=.4)` | **ต้องมี** (cutoff=1) |
| Province (Thai/Lao) | **ไม่มี** — eval path = letterbox + ToTensor + Normalize เท่านั้น | `RandomAutocontrast(p=.4/.5)` | **ไม่ต้องมี** — autocontrast ที่เป็น augmentation เป็นเพียง invariance ⇒ **ห้ามใส่เพิ่มที่ serve** (จะเลื่อน distribution) |
| M1/M2/M3A detectors | ไม่มีเรื่องสี (LibreYOLO/RF-DETR จัดการเอง) | Roboflow ของผู้ให้ข้อมูล | แค่ size/normalization |

**คำตอบสั้น:** contrast augmentation **ไม่จำเป็นกับทุกโมเดล** — เลือกใส่เฉพาะที่มันอยู่ในพาธ deterministic (char/OCR) ส่วน province ไม่ต้อง ⇒ ไม่มีเหตุให้ "serve ต้องทำ preprocessing เหมือนกันหมดทั้งกระดาน"

### แล้ว "หยิบโมเดลมาต่อกัน กำหนดแค่ size" พอไหม — วัดแล้ว

โหมด `I_chain_no_heuristics` = คงสัญญา geometry/สีครบทุกข้อ (deskew, letterbox pad, grayscale, autocontrast) แต่ปิด heuristic ทั้งหมด (CLAHE, stroke, regex repair, CTC fallback, reconciliation):

| mode | province (120 ภาพ) | detected | plate_text |
|---|---|---|---|
| `A_current` (ครบทุกอย่าง) | **112/120 = 93.3%** | 118 | — |
| `I_chain_no_heuristics` (ต่อโมเดลเฉยๆ) | **112/120 = 93.3%** | 118 | **เหมือนกันแค่ 84/120** |

⇒ **"กำหนด size/geometry ให้ถูก" พอสำหรับจังหวัด (เท่ากันเป๊ะ 120/120) แต่ไม่พอสำหรับตัวอักษร** — ต่างกัน 36/120 ภาพ และที่หายไปคือของพัง เช่น `70-5742 → 7054` (หายตัวอักษร), `6กร 913 → 6กร 93`, `ฎห 2635 → ผบ 2635` (ตัวอักษรผิด), `งน 2297 → ง112297` (มีขยะแทรก)

**แนวปฏิบัติ:** ใช้ "ต่อโมเดล + size ถูก" เป็น baseline ที่ดีและตรวจสอบง่าย แต่ต้องมีชั้น post-processing สำหรับสตรีมตัวอักษร (NMS + split wide box + regex guard + box/CTC fusion) ถ้าต้องการข้อความที่ใช้งานได้จริง
