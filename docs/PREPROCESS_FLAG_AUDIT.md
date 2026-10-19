# PREPROCESS FLAG AUDIT — เปิด OpenCV แล้วทำไมทายผิด / OpenCV มีผลจริงตรงไหน

> วันที่: 2026-10-02 · เทียบด้วย `scratch/ab_flag_matrix.py` (โมเดลชุดเดิม โหลดครั้งเดียว สลับ flag ทีละตัว)
> ชุดทดสอบ: 120 ภาพจาก `thai-car-license-plate-province.v5i.yolov11/valid` (สุ่มแบบ stratified ทุกๆ 4 ภาพจาก 481)
> GT ที่มี: **province เท่านั้น** — v5i ไม่มี char-level GT

---

## 0. สรุปสั้น (TL;DR)

1. **อาการ "เปิด OpenCV ปรับภาพตรงแล้วผิดเยอะกว่าภาพเอียง" ไม่ reproduce** — deskew ON = 87.5% (105/120), deskew OFF = 86.7% (104/120) ต่างกัน 1 ภาพ = noise และ geometry ที่โมเดลเห็นแทบไม่ต่างกันเลย (`plate_frac` 0.856 vs 0.837 เทียบกับค่าจริงของ train = 0.841)
2. **ตัวการจริงคือ `ENABLE_AMBIENT_CORNER_PAD = False`** → **−5.8 คะแนน** (87.5% → **93.3%**) เพราะมัน **พังสัญญา train/serve ของ province** โดยตรง และเปิดกลับแล้ว **ข้อความเพี้ยน 0/120 ภาพ** (plate_text เหมือนเดิมทั้ง 120)
3. `ENABLE_AUTOCONTRAST` **ไม่มีผลต่อ province เลย** (0/120 เปลี่ยน) เพราะมันแตะเฉพาะ crop ตัวอักษร ⇒ ตั้ง True ถูกแล้ว (ตรงกับตอน harvest)
4. `ENABLE_GRAYSCALE_PROV_NORM = False` แย่ลง −0.8 ⇒ ต้องเป็น True
5. OpenCV + heuristic ทั้งชุด (`PURE_MODEL_MODE=True`) ทำให้ province **แย่ลงแค่ 0.8** แต่ทำ **ข้อความพัง 50/120** ⇒ heuristic จำเป็นกับ "ตัวอักษร" ไม่ใช่กับ "จังหวัด"

---

## 1. ตารางผลวัด (รันเดียวกันทุกประการ ต่างแค่ flag)

| mode | เปลี่ยน flag อะไร | province acc | detected | ใช้ corner quad | `plate_frac` median | plate_text |
|---|---|---|---|---|---|---|
| **A (เดิม)** | ค่าที่ส่งมา (pad=False) | 105/120 = **87.5%** | 117 | 89 | 0.856 | — |
| **A (ใหม่)** | `ENABLE_AMBIENT_CORNER_PAD=True` | 112/120 = **93.3%** | 118 | 90 | 0.853 | เหมือน C ทุกภาพ |
| B | `HOMOGRAPHY_DESKEW=False` | 104/120 = 86.7% | 116 | 0 | 0.837 | ต่าง 33 ภาพ |
| C | `AMBIENT_CORNER_PAD=True` | 112/120 = **93.3%** | 118 | 90 | 0.853 | ต่างจาก A(เดิม) 12 ภาพ |
| D | `AUTOCONTRAST=False` | 105/120 = 87.5% | 117 | 89 | 0.856 | ต่าง 9 ภาพ |
| E | `GRAYSCALE_PROV_NORM=False` | 104/120 = 86.7% | 116 | 88 | 0.857 | ต่าง 1 ภาพ |
| G | `PAD=True` + `DESKEW=False` | 112/120 = **93.3%** | 117 | 0 | 0.837 | ต่างจาก C 29 ภาพ |
| H | `PURE_MODEL_MODE=True` (ปิดทุกอย่าง) | 104/120 = 86.7% | 113 | 0 | 0.844 | ต่างจาก A(เดิม) **50 ภาพ** |

**ค่าอ้างอิงจากข้อมูล train จริง** (`v5i/train` 600 ภาพ): median **192×99 (aspect 1.94)**, `plate_frac` median **0.841**
⇒ ทั้ง deskew ON (0.856) และ OFF (0.837) ใกล้ค่าอ้างอิงมาก ⇒ **warp ไม่ได้ทำให้ crop หลุดไปจาก distribution ที่โมเดลคุ้นเคย**

---

## 2. Root cause #1 — `ENABLE_AMBIENT_CORNER_PAD` พังสัญญา train/serve ของ province

**ทำไมถึงพัง:** มี `GrayscaleSmartResize` สองตัวที่ควรจะเหมือนกันแต่คิดไม่เหมือนกัน

| ที่ไหน | พฤติกรรม | toggle? |
|---|---|---|
| `src/train_grayscale_province_thai.py:41` (class ส่วนตัว) | letterbox ไป `(256, 80)` เติมขอบด้วย **สีพื้นหลัง = median ของ 4 มุม** | **ไม่มี** ทำแบบนี้เสมอ |
| `src/preprocess.py:93` `GrayscaleSmartResize` (ที่ serve ใช้) | อ่าน `Config.is_ambient_padding_enabled()` | **มี** ← flag ที่ปิดอยู่ |

⇒ ตอน `False` serve จะ **stretch ตรงๆ** แทน letterbox:

```
crop จริงของ province (วัดจาก province_crops_v2/train, n=3000)
   median 118 × 30  → aspect 4.24
tensor ที่โมเดลต้องการ 256 × 80 → aspect 3.20
   letterbox : คง aspect ไว้ เติมขอบบน-ล่างด้วยโทนพื้นจาน
   stretch   : บีบแนวนอน 1.33 เท่า  ← โมเดลไม่เคยเห็น
```

**flag ตัวเดียวกันยังคุมอะไรอีก**
- `api_server.py:2178` การ pad crop ตัวอักษรเป็น **square** ก่อนเข้า `tf_char` (ซึ่งเป็น `Resize((64,64))`) — และ char crop ตอน harvest ถูก pad เป็น 64×64 มาก่อนแล้วด้วย `extract_recognition_crops_v2.pad_to_square` ⇒ ถ้าปิด flag นี้ crop ที่ไม่จัตุรัสจะถูก **ยืด** แทนที่จะ pad ⇒ mismatch อีกจุด
- `api_server.py:1487` การ pad digit ใน truck banner (แต่ path นี้ตายอยู่แล้วเมื่อ `ENABLE_STROKE_DISAMBIGUATION=False`)

**ทำไม test เดิมจับไม่ได้:** `src/test_classifier_preprocess.py` เช็คแค่ว่า "มุมภาพมีโทนพื้น ไม่ใช่ดำ" — แต่ crop สังเคราะห์มีพื้นหลังสีเดียวทั้งภาพ ⇒ ทั้ง letterbox และ stretch ให้ค่ามุมเท่ากันและผ่านทั้งคู่ ⇒ **test เขียวทั้งที่ flag ผิด** → แก้แล้วในข้อ 6

---

## 3. Root cause #2 (เชิงโครงสร้าง) — สัญญาเดียวกันถูก implement สองที่

trainer ไม่ import `src/preprocess.py` แต่ **ก็อปคลาสมาไว้เอง** ⇒ แก้ที่หนึ่ง อีกที่หนึ่งเพี้ยนเงียบๆ ได้ (ซึ่งคือสิ่งที่เกิดขึ้นจริง)

ความต่างที่พบระหว่าง trainer กับ shared transform (โหมด train):

| augmentation | trainer `train_grayscale_province_thai.py:66` | `preprocess.get_grayscale_prov_transforms(is_train=True)` |
|---|---|---|
| affine | `RandomAffine(5, translate=(.03,.04))` | `RandomAffine(4, translate=(.02,.04))` |
| sharpness | `RandomAdjustSharpness(2.0, p=.5)` | **ไม่มี** |
| autocontrast | `p=0.5` | `p=0.4` |
| resize/pad | letterbox เสมอ | ตาม config ← จุดที่พัง |

⇒ ถ้ามีใคร retrain ผ่าน shared transform จะได้ augmentation คนละชุดกับที่โมเดล v5 ถูกเทรนมา

---

## 4. คำถาม: deskew ON ดีกว่าหรือแย่กว่า OFF

**โค้ดทำอะไรจริง** (`api_server.py:1778-1836`)

- M1 ปัจจุบัน = `plate_detector_picodet_s_v2.pt` และ `LibreYOLOWrapper` **บังคับ `r.masks = None`** (D-FINE/PicoDet ไม่ใช่ segmentation) ⇒ กิ่ง `res1.masks.xy` **ไม่เคยรัน**
- ⇒ deskew ON จริงๆ คือ: crop bbox + pad 12% → ย่อเป็น 224×224 → ให้ **`plate_corner_regressor.pth` ทาย 4 มุม** → `warp_perspective_plate(padding_frac=0.06)` → `fine_deskew_plate` (ค้นมุมเอียง ±8° แบบ brute force)
- ถ้าทายมุมไม่ได้ → ตกไปใช้ bbox crop เหมือนโหมด OFF
- **วัดได้: มี quad จริงแค่ 90/120 ภาพ** ⇒ "deskew ON" ไม่ใช่ transform เดียว แต่เป็น **ส่วนผสม** (25% ของภาพยังได้ bbox ธรรมดา)

**ผลที่วัดได้: เสมอกัน** — 87.5% vs 86.7%, province ต่างกัน 9 ภาพ (กระจายสองทาง), `plate_frac` 0.856 vs 0.837
**แต่ข้อความต่างกัน 33 ภาพ** ⇒ warp ไม่ได้ทำให้ "ถูกขึ้น/ผิดลง" เป็นระบบ มันแค่ **เพิ่ม variance**

**ประเด็นสำคัญที่สุด — warp ไม่เคยถูกใช้ตอนเทรน**

```
grep -rn "warp_perspective_plate|extract_quad_corners" src/*.py
  → src/api_server.py            (serve)
  → src/harvest_lao_province_candidates.py   (harvest ลาว)
  → src/prepare_ground_truth_crops.py        (GT crops — โฟลเดอร์ output/ground_truth_crops ลบไปแล้ว)
  → src/prepare_from_yolo.py
  ✗ ไม่มี training script ของ M2 / M3A / M3B ใช้เลย
```

crop ที่ใช้เทรน v2 (province/char) มาจาก `extract_recognition_crops_v2.py` ซึ่งเริ่มจากภาพ **192×99 ที่ rectified มาแล้วโดย Roboflow** แล้วตัดด้วย **bbox inset** ตามกติกา `api_server.py:1796-1813` — เอกสารหัวไฟล์เขียนไว้ชัดว่า "กัน train/serve mismatch"
⇒ **homography เป็น transform ที่มีอยู่แค่ตอน serve** มันบังเอิญตกลงใกล้พอ (0.853 vs 0.841) จน province ไม่พัง แต่การที่มันเป็น transform ที่ไม่มีในข้อมูลเทรนคือคำอธิบายว่าทำไม variance ของตัวอักษรสูง

> หมายเหตุ: `train_country_classifier.py` (M1.5) ยอมรับ `output/ground_truth_crops/rectified_plates` ซึ่ง `prepare_ground_truth_crops.py` สร้างด้วย `warp_perspective_plate` ⇒ **M1.5 อย่างน้อยก็ compatible กับภาพที่ warp แล้ว** ขณะที่ M3B ไม่ใช่ — สัญญาไม่สม่ำเสมอในโปรเจกต์เดียวกัน

---

## 5. คำถาม: "ถ้าทำตาม workflow M1 AABB → warp → M1.5 → M2 → M3 OpenCV ควรไม่มีผลแล้วไม่ใช่หรือ"

**ถูกครึ่งหนึ่ง** — และโค้ดก็ทำตามนั้นแล้ว:

- หลัง warp จบ **ไม่มี cv2 ทำ geometry อีกเลย** ในเส้นทาง M1.5 → M2 → M3A → M3B ตำแหน่ง box คำนวณจากสัดส่วน (inset `0.03*rw`, `0.46*rh` ฯลฯ) แล้วส่งเข้า transform ของแต่ละโมเดล
- แต่ที่เหลือใน OpenCV **ไม่ใช่เรื่อง perspective** เลย:

| กลุ่มงาน | ที่อยู่ | จำเป็นไหม |
|---|---|---|
| กู้ภาพตอน M1 พลาด (CLAHE, ยิงซ้ำที่ conf 0.20) | `api_server.py:1648` | เฉพาะภาพมืด/จานเล็ก |
| crop inset + square pad + letterbox ให้ตรงกับ dataset | `api_server.py:1487,2178` + `preprocess.py` | **จำเป็นมาก** — นี่คือ pixel contract |
| post-process เชิงเรขาคณิตบนตัวอักษร (findContours, apex, split stroke) | `api_server.py:2198,2216` | ช่วยเฉพาะตัวอักษร |
| autocontrast | `api_server.py:2170` | จำเป็น (ตรงกับ harvest) |

⇒ คำตอบคือ: **OpenCV ไม่จำเป็นในแง่ "แก้ภาพเอียง" (warp ตัวเดียวพอ) แต่จำเป็นในแง่ "ทำให้ pixel ตรงกับตอนเทรน"**
ถ้าอยากให้ข้อสงสัยนี้หมดไปจริงๆ ต้อง **ใช้ warp ตอน harvest ด้วย** (re-extract v2 crops ให้ผ่าน `warp_perspective_plate` ตัวเดียวกับ serve) แล้วมันจะกลายเป็น transform ที่โมเดลรู้จัก

---

## 6. สิ่งที่แก้ไปในรอบนี้

1. **`src/config.py`** — `ENABLE_AMBIENT_CORNER_PAD: False → True` พร้อมคอมเมนต์ระบุหลักฐานและตัวเลข (เพื่อไม่ให้มีใครปิดอีกโดยไม่รู้)
2. **`src/test_classifier_preprocess.py`** — เพิ่มข้อ 5 **aspect guard**: ยัด crop จตุรัส 60×60 (aspect 1.0) เข้า `GrayscaleSmartResize((256,80))` แล้ววัดความกว้างของ glyph
   - letterbox → ~25 px, stretch → ~83 px
   - เดิม test เขียวทั้งที่ flag ผิด / ตอนนี้ **FAIL ทันทีถ้าตั้ง flag เป็น False** (ยืนยันแล้วทั้งสองทาง: 17 PASS กับ default, 16 PASS 1 FAIL เมื่อบังคับ False)

**ยืนยันด้วยการรันซ้ำ:** หลังแก้ config แล้วรัน mode A ใหม่ทั้ง 120 ภาพ → **112/120 = 93.3%** และ **province + plate_text เหมือน mode C ทุกภาพ (120/120)** ⇒ การเปลี่ยนแปลงนี้ให้ผลเท่ากับเปิด ambient pad เป๊ะ และไม่กระทบตัวอักษรเลย

### ค่าที่แนะนำทั้งบล็อก

```python
ENABLE_OPENCV_CLAHE = False           # ทำงานเฉพาะตอน M1 ไม่เจอจาน — ไม่มีเหตุเปิดใน production ปกติ
ENABLE_AMBIENT_CORNER_PAD = True      # ★ แก้จาก False — สัญญา train/serve ของ province (+5.8 คะแนน)
ENABLE_GRAYSCALE_PROV_NORM = True     # ★ ต้อง True (−0.8 ถ้าปิด)
ENABLE_STROKE_DISAMBIGUATION = False  # ★ ระวัง: flag นี้ยังคุม truck banner ด้วย (ดูข้อ 7)
ENABLE_HOMOGRAPHY_DESKEW = True       # เป็นกลางต่อ province (87.5 vs 86.7) — รอ char-level GT ค่อยตัดสิน
ENABLE_AUTOCONTRAST = True            # ★ ต้อง True (ตรงกับ extract_recognition_crops_v2.pad_to_square)
ENABLE_REGEX_REPAIR = True            # ★ ปิดแล้วข้อความพังยับ (ดู H: 'กก1กกก1', '777พ3020')
ENABLE_FALLBACK = True                # ★ เหตุผลเดียวกับข้างบน
ENABLE_RECONCILIATION = True          # ★ เหตุผลเดียวกับข้างบน
```

---

## 7. ข้อบกพร่องที่เจอระหว่างทาง (ยังไม่ได้แก้)

1. **`ENABLE_STROKE_DISAMBIGUATION` ผูกกับ truck banner** — `api_server.py:1470` (`extract_dlt_truck_code`) `if cfg.PURE_MODEL_MODE or not cfg.USE_STROKE_DISAMBIGUATION: return None, None, 0.0, False` ⇒ ตั้งใจปิดการแก้ตัวอักษร แต่ดันปิดการอ่านรหัสจังหวัดบนป้ายรถบรรทุกไปด้วย ทั้งที่ไม่เกี่ยวกัน
2. **`PURE_MODEL_MODE` ไม่ใช่ superset ที่สะอาด** — มันปิด deskew ด้วย ⇒ ในการทดลองจะแยกผลของ CLAHE / stroke / regex ออกจากกันไม่ได้ด้วยสวิตช์เดียว (mode H จึงได้ `quad=0`)
3. **property ซ้ำซ้อนสองชุด** — `cfg.USE_X` (instance) กับ `Config.is_x_enabled()` (classmethod) ให้ค่าเดียวกัน ⇒ เสี่ยงแก้ไม่ครบคู่
4. **augmentation ของ trainer ไม่ตรงกับ shared transform** (ตารางข้อ 3)

---

## 8. ข้อจำกัดของข้อสรุปนี้

- **ไม่มี char-level GT บน v5i** ⇒ ประเมินผลกับ "ตัวอักษร" ได้แค่นับว่า prediction เปลี่ยนไปกี่ภาพ ไม่รู้ว่าถูกหรือผิด ⇒ ข้อสรุปเรื่อง deskew ที่ว่า "เสมอกัน" เป็นจริงในมุม province เท่านั้น
- 120 ภาพ → margin ประมาณ ±4% ⇒ **ตัวเลข −5.8 คะแนนชัดเจนพอ** แต่ความต่าง 87.5 vs 86.7 (1 ภาพ) ยังสรุปเป็นนัยสำคัญไม่ได้
- **192×99 เป็น domain ที่ char model ไม่เคยเห็น** (crop ตอนเทรนใหญ่กว่านี้) ⇒ ตัวเลข plate_text ที่เห็นใช้ตัดสินคุณภาพ char model ไม่ได้
- ที่ความละเอียดจริง (production) bbox กับ quad อาจต่างกันมากกว่าที่ 192×99 ซึ่ง bbox แนบพอดีจานอยู่แล้ว
- ยังไม่ได้ยืนยันว่า v5 เทรนด้วย crop ชุดใดเป๊ะๆ (`output/ground_truth_crops` ถูกลบแล้ว) — ข้อสรุปเรื่อง "warp ไม่เคยอยู่ในข้อมูลเทรน" มาจากการที่ไม่มี training script ใด import `warp_perspective_plate` + docstring ของ extractor

---

## 9. วิธีรันซ้ำ

```bash
cd <repo>
# A/B ทุก flag (resumable, เขียน scratch/ab_flag_matrix_results.csv)
FORCE_CPU=True AB_N=120 /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python -u scratch/ab_flag_matrix.py

# เอาเฉพาะบาง mode
FORCE_CPU=True AB_N=120 AB_MODES=A_current,D_autocon_off ... python -u scratch/ab_flag_matrix.py

# contract test (ตอนนี้ต้อง 17 PASS 0 FAIL)
FORCE_CPU=True /Users/kwankhaos/miniconda3/envs/thai-lpr/bin/python src/test_classifier_preprocess.py
```
