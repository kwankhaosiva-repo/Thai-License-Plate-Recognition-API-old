# บทวิเคราะห์ + แผนปรับปรุง v6 (อิงหลักฐานที่วัดจริง — 2026-09-29)

> เอกสารนี้ตอบคำถาม: *"ควร train เพิ่มโดยปรับ dataset / serve pre-process / post-process / augmentation / auto-contrast / size / scale อย่างไร ระหว่าง train, pre-train, pre-processing และการตัดสินใจใน post-processing"*
> ทุกตัวเลขในเอกสารนี้**วัดจาก artifact จริงในเครื่อง** ไม่ใช่การประมาณ — ระบุไฟล์อ้างอิงไว้ทุกจุด

---

## 0. TL;DR — ลำดับความสำคัญ

| ลำดับ | สาเหตุราก | หลักฐานชี้ขาด | สิ่งที่ต้องทำ |
| :--- | :--- | :--- | :--- |
| **P0** | **Province class จำนวนมากมี "ภาพต้นทางจริง" แค่ 1–5 ภาพ** (crop เยอะเพราะ aug/copy) | 6 คลาสมี unique source = **1 ภาพ** / 31 crops (เช่น น่าน, พังงา, อุตรดิตถ์, นครพนม, ปัตตานี, แม่ฮ่องสอน) → 36/77 คลาสมี ≤5 sources | หา**ภาพจริงใหม่** ต่อคลาส (เป้า ≥30 unique source/คลาส) + ยืนยันด้วย pHash |
| **P0** | **Resolution domain gap**: 71% ของ province train crop มาจากภาพ 192px (native 61×29 → ถูก upscale 4.2×) แต่ serve เห็น crop คมระดับ native | province crops: h5+plain = 5,547/7,804 = 71% มี native width **61px** (≈6 px/ตัวอักษร); h4 = 141px | harvest จากภาพ **full-res เท่านั้น** + ใส่ **resolution-degradation augmentation** ตอน train |
| **P1** | ใน-dataset 100% เป็นเลขที่**มองโลกในแง่ดีเกินจริง** | province train/valid แชร์ "ภาพต้นทาง" กัน **62/77 คลาส (81%)**; char **46/48 (96%)** | re-split แบบ **source-disjoint** แล้ววัดใหม่ก่อนเชื่อตัวเลข 100% |
| **P1** | **Input เล็ก → M1 หาแผ่นป้ายไม่เจอเลย** | 14/39 error = `detected=False` และ **ทั้ง 14 ภาพเป็น 192×99**; test-time upscale 3× → กู้คืน detection ได้ **8/14** และทายจังหวัดถูกล้วน 5 ภาพ (91.89% → **92.93%**) | gate ความละเอียดขั้นต่ำ + upscale ก่อน M1 เมื่อ input เล็ก |
| **P2** | **class ข้อมูลน้อยกลายเป็น "ตัวดูด" (attractor)** | ทาย **ระนอง 12 ครั้ง** ใน 481 ภาพ ทั้งที่ GT ระนอง = **0** และคลาสนี้มี unique source แค่ 3 ภาพ | เพิ่มคลาส **unknown/unreadable** + logit/prior correction สำหรับคลาสที่ sources < 5 |
| **P2** | ตัวอักษรเดี่ยวผิด 1 ตัว = ป้ายผิดทั้งป้าย (last mile) | 181/374 ป้ายผิด → **98 ป้าย (54%) ผิดตัวเดียว**, 72 ผิดหลายตัว, 11 ได้ข้อความว่าง | beam/top-k + ใช้ `alt_candidates` เป็น candidate ให้ validator เลือก + retrain OCR |
| **P2** | **GT ของ whole-plate เป็น pseudo-label** → ตัวเลข 51.60% ปน noise ของ GT เอง | `full_ocr_text` มาจาก `candidates_metadata.csv` ที่เป็น output ของโมเดล (มี `confidence` คอลัมน์, ชื่อไฟล์ `auto_*`) เช่น GT `ฆช 8734` (ฆ แทบไม่ปรากฏบนป้ายจริง = GT ผิด) | สร้างชุด **human-verified 200–300 ป้าย** เป็น headline metric |

**ตัวเลข headline ที่ควรใช้จริงตอนนี้:** *บนภาพ 192×99 (stress test) province = 91.89% (442/481), whole-plate = 51.60% (193/374), char recall = 84.88% (1,908/2,248) — และสามตัวนี้ยังไม่ได้หัก GT noise.* ยัง**ไม่มี**ตัวเลข production-resolution ที่เชื่อถือได้ → งาน E1 ด้านล่างคือสิ่งที่ควรทำก่อนตัดสินใจเรื่อง config/โมเดล

---

## 1. หลักฐานที่วัดได้ (ฐานของทุกข้อเสนอ)

| # | สิ่งที่วัด | ผลที่ได้ | ไฟล์อ้างอิง |
| :--- | :--- | :--- | :--- |
| 1 | In-dataset: char v5 / province v5 (R18, R34) | char Top-1 **99.54%** (n=3,511), province **100.00%** (n=1,285) ทั้ง R18 และ R34 | `scratch/eval_v4_all.py` |
| 2 | End-to-end บน v5i valid 481 ภาพ | province **91.89%** (442/481) | `scratch/eval_v5i_e2e_results.csv` |
| 3 | End-to-end 374 ภาพที่มี GT ข้อความ | province 92.78% (347/374), whole-plate **51.60%** (193/374), char recall **84.88%** (1,908/2,248) | `scratch/eval_v5i_model3_results.csv` |
| 4 | โครงสร้างของ 39 error (481 ภาพ) | **14** = M1 หาป้ายไม่เจอ (`detected=False`, ทั้งหมด 192×99) · **2** = ทายเป็นประเทศลาว · **12** = ทาย `ระนอง` · **11** = จังหวัดไทยอื่นผิด | `scratch/eval_v5i_e2e_results.csv` |
| 5 | จำนวน "ภาพต้นทาง" จริงต่อคลาสจังหวัด | **6 คลาส = 1 source** (31 crops), 36/77 คลาส ≤5 sources, 38/77 <10 sources | `datasets/Thai/recognition_v2/province_crops_v2/train` |
| 6 | Train/valid แชร์ภาพต้นทาง | province **62/77 คลาส (81%)** (กทม. แชร์ 322 ภาพ), char **46/48 (96%)** | เทียบ stem ระหว่าง `train/` กับ `valid/` |
| 7 | ความละเอียดของข้อมูล train vs eval | v5i **97% เป็น 192×99** (ทั้ง train 2,349 และ valid 481), province crop native median **61×29** (h5/plain 71%) vs h4 **141×32**, transform `GrayscaleSmartResize(256,80)` = upscale **4.2×** | `datasets/Thai/thai-car-license-plate-province.v5i.yolov11/`, `province_crops_v2/` |
| 8 | Char train long tail | 27,395 crops / 48 คลาส, บางสุด: ป 52 · ด 57 · ฬ 59 · ฟ 60 · ฐ 63 · ภ 63 (ห่างจาก 7 = 2,797 → 54:1) | `char_crops_v2/train` |
| 9 | Augmentation ที่ใช้อยู่ | province: Affine 5°, Jitter 0.3, Sharpness 2.0 p=.5, Autocontrast p=.5 — **ไม่มี blur / ไม่มีการลด-เพิ่ม scale / ไม่มี perspective** · char: Affine 6° scale .95–1.05, Jitter .35, Gray .15, Perspective .12 — **ไม่มี blur/scale** | `src/preprocess.py`, `train_grayscale_province_thai.py`, `train_character_classifier.py` |
| 10 | Confusion ตัวอักษรที่พบบ่อย (gt→pred) | `7→1` ×9 · `ฆ→ข` ×8 · `ฉ→ค` ×5 · `ร→ว` ×4 · `7→ษ` ×3 · `1→ข` ×3 · `2→ก` ×2 | `scratch/eval_v5i_model3_results.csv` |
| 11 | ทดลอง test-time upscale 3× (LANCZOS4) ก่อน M1 | กับ 14 ภาพที่ M1 มองไม่เห็น: **กู้คืน detection 8/14**, ทายจังหวัดถูกเพิ่ม **5 ภาพ** (⊂ นั้นมี 2 ภาพเปลี่ยนไปทาย `ระนอง`) | รันจริง 2026-09-29 (บทใน section 3.4) |
| 12 | บั๊ก fusion ป้ายซ้ำ | `2ถล 1238` (box มี `ล` เกิน 1 ตัว) → เดิมได้ `2ถลล1238` (Custom) → ตอนนี้ได้ `2ถล 1238` (NCC NNNN) ✅ และ `5ศ 7856` ยังคงเป็น `5ศ 7856` (NC NNNN Trailer) ✅ | `align_and_fuse_thai_sequences` + letter-plate guard ใน `src/api_server.py` |

---

## 2. ข้อวินิจฉัย — คอขวดจริงอยู่ตรงไหน

1. **ไม่ใช่ตัว classifier ในโดเมนของตัวเอง** — char 99.54% / province 100% (แต่เป็นตัวเลขในโดเมนที่ leak ซึ่งเป็นข้อ 6)
2. **จังหวัดคือปัญหาที่ "ข้อมูล" ไม่ใช่ที่ "โมเดล"** — คลาสที่มี unique source 1–5 ภาพไม่สามารถ generalize ได้: โมเดลเรียน "รูป 1 รูป + aug 30 แบบ" ไม่ใช่ "ป้ายจังหวัดนั้น" → เกิดทั้งการทายผิดและ attractor `ระนอง`
3. **ความละเอียดคือปัญหาที่สอง** — train ถูกป้อนข้อความจังหวัดที่มีข้อมูลจริง ~6 px/ตัวอักษร (หลัง upscale 4.2×) แต่ serve เห็นภาพคมของจริง → เป็น domain gap สวนทางกับที่คิด (train เบลอ, serve คม) และเป็นคำอธิบายที่น่าจะถูกที่สุดของ "R18 ดีกว่า R34" ที่เจอบน serve: R18 มี capacity น้อยกว่า จึงเกาะรูปทรงหยาบ (ซึ่งเป็น signal เดียวที่มีในภาพ 6 px) ขณะที่ R34 มี capacity พอจะ fit artifact ความถี่สูงของภาพ thumbnail ซึ่ง transfer ไปภาพคมได้แย่กว่า — **ยังต้องพิสูจน์ด้วย E1**
4. **Detection ที่ input เล็กคือความผิดพลาดที่แก้ได้ทันที** — 14/481 = 2.9% ของ error ทั้งหมด และเกือบทั้งหมดกู้คืนได้ด้วย upscale (ข้อ 11)
5. **Post-processing ยัง "บังคับตอบ"** — เมื่อไม่มั่นใจจริง โมเดลยังต้องเลือก 1 ใน 77 และระบบยัง emit ออกมา (มีตัวอย่าง `province_classification = 0.384` ที่ยังส่งชื่อจังหวัดออก) — ควรมีทาง "ไม่ตอบ"

---

## 3. แผนปรับปรุงแยกตาม stage

### 3.1 Dataset & pre-train (harvest) — P0

**เปลี่ยน "ตัวชี้วัด" ของ dataset:** เลิกนับ *จำนวน crop ต่อคลาส* (ซึ่งถูก aug หลอกได้) ให้นับ **จำนวนภาพต้นทางจริง (unique source) ต่อคลาส** และเก็บลง manifest

- **เป้าหมาย:** province ≥ **30 unique source/คลาส** (ตอนนี้ต่ำสุด 1), char ≥ **25 unique source/คลาส**
- **ปริมาณที่ต้องหาเพิ่ม (ขั้นต่ำสุดเพื่อปิดรู):** 36 คลาสที่มี ≤5 sources × ~25 ภาพ ≈ **~900–1,200 ภาพจริง** — เน้น 12 คลาสที่แย่สุด (อุตรดิตถ์, พังงา, น่าน, นครพนม, ปัตตานี, แม่ฮ่องสอน, ร้อยเอ็ด, แพร่, สตูล, พัทลุง, กาฬสินธุ์, นราธิวาส)
- **คุณภาพการ harvest (ใหม่):**
  - รับเฉพาะภาพต้นทาง **≥ 1000 px** ด้านยาว; ปฏิเสธ crop ที่ province strip native < 96 px (≈<8 px/ตัวอักษร) และ char crop native < 24 px
  - บันทึก `native_w`, `native_h`, `blur_var`, `source_hash` ต่อ crop ลง manifest → ใช้ตรวจ/กรองภายหลังได้
  - **pHash dedupe ที่ระดับ source** (ตอนนี้ `harvest_v5_c0/c1/c2` มี crop ซ้ำจากภาพเดียวกันปนอยู่)
- **Re-split แบบ source-disjoint จริง** (group by source image **และ** session/ผู้โพสต์) — ต้องทำก่อนวัดผล เพราะตอนนี้ 81%/96% แชร์ source กัน (ข้อ 6)
- **เพิ่มคลาสสำหรับ "สิ่งที่ควรถูกปฏิเสธ"** (สำคัญและได้ผลเร็ว)
  - char: เพิ่มคลาส **`junk/background`** (เศษขอบป้าย, dash, สกรู, แถบสี) — ปัจจุบัน classifier 50 คลาสถูกบังคับให้ map junk เป็นตัวอักษรจริง ต้นเหตุของ `7→ษ`, `1→ข`, `2→ก` (ข้อ 10) และของ `5`/`4` โผล่มาเกิน
  - province: เพิ่มคลาส **`unknown/unreadable`** จาก strip ที่เบลอ/ถูกบัง/เพี้ยน — ให้โมเดลมีทาง "ไม่รู้" แทนการถูกบังคับให้เลือก `ระนอง`

### 3.2 Augmentation (train) — P1

ของเดิม (Affine/Jitter/Sharpness/Autocontrast) ไม่แตะ — **เพิ่มแกนที่ยังขาด** ซึ่งตรงกับ gap ที่วัดได้ข้อ 7:

| แกนที่เพิ่ม | พารามิเตอร์ที่เสนอ | เหตุผล (ผูกกับหลักฐาน) |
| :--- | :--- | :--- |
| **Resolution degradation** (สำคัญสุด) | `RandomApply([Downscale(s=0.15–0.5, INTER_AREA) → resample กลับขนาด contract], p=0.5)` | จำลองทั้ง 2 โลก (6 px/ตัวอักษร และภาพคม) → ทำให้ train/serve ไม่ผูกกับความละเอียดใด ๆ |
| Gaussian blur | `radius 0.5–2.0, p=0.4` | ภาพจริงจากกล้องวงจรปิด/ระยะไกล |
| Motion blur | `p=0.15` (kernel 3–9, มุมสุ่ม) | ภาพถ่ายรถที่กำลังเคลื่อนที่ |
| JPEG artifact | `quality 30–70, p=0.3` | คลิปจากกล้อง/วิดีโอที่ถูกบีบอัด |
| Specular glare / shadow | `RandomShadow / highlight blend, p=0.2` | ป้ายพลาสติกสะท้อนแสง (เคสที่ผู้ใช้เจอ) |
| RandomErasing (occlusion) | `p=0.15, scale 2–8%` | สกรู/กรอบบังตัวอักษร |
| Scale ใน affine ของ province | เปลี่ยน `scale=(0.8, 1.2)` + เพิ่ม `RandomPerspective(0.1, p=0.3)` | ให้ทนต่อการ rectify ที่ไม่เป๊ะ (เคส `83-4203`) |

**ข้อควรระวัง:** อย่าลบ augmentation เดิม และอย่า augment ที่ native ความละเอียดต่ำแล้ว upscale ทีหลัง (จะไม่ได้ผล) — ให้ augment **หลัง** downscale step เสมอ

### 3.3 Auto-contrast — P1 (ต้องทดลอง A/B ก่อนแก้)

สถานะปัจจุบัน: serve ใช้ `autocontrast(cutoff=2)` (char) และ province ไม่มี autocontrast ใน serve แต่ train มี `RandomAutocontrast(p=0.5)` → **ไม่สมมาตรกัน** ตาม contract ที่ควรเป็น

- **ทดลอง A/B บนภาพจริง (E1):** `cutoff ∈ {none, 1, 2}` × {char, province} → วัด Top-1 ต่อตัว
- **ความเสี่ยงที่ต้องรู้:** autocontrast เป็น operation **global per-crop** ถ้า crop ติดกรอบขาวของป้ายเข้ามาด้วย มันจะ normalize ให้กรอบขาวกลายเป็น "ขีดสุด" แล้วตัวอักษรจม → ตัวอักษรที่อ่านได้กลับอ่านไม่ได้ (ตรงกับอาการในภาพที่ 3 ของผู้ใช้)
- **ทางเลือกที่ควรพิจารณา:** CLAHE (`clipLimit 2.0, tile 4×4`) บน crop แทน autocontrast global, หรือ autocontrast แบบ **masked** (คำนวณเฉพาะ ROI ของตัวอักษร)
- **ข้อบังคับ:** ต้องเลือกแบบเดียวแล้ว **เทรนใหม่ให้ตรงกับ serve 100%** (ห้าม serve อย่าง train อย่าง) — ถ้าเลือก CLAHE ต้องใส่ CLAHE ใน transform ฝั่ง train ด้วย

### 3.4 Serve pre-processing — P1

1. **Resolution gate + upscale-for-detection** (วัดแล้วในข้อ 11):
   ```
   ถ้า max(w,h) < 640: ย่อ/ขยาย 3× ด้วย INTER_LANCZOS4 ก่อนส่ง M1
   ```
   ผลที่วัดได้: detection กลับมา **8/14** ภาพที่เคยพลาด และจังหวัดถูกเพิ่ม **5 ภาพ** → 91.89% → **92.93%** บนชุด 481 (ต้นทุน compute จิ๋ว เพราะเฉพาะภาพเล็ก)
2. **Plate-crop minimum-resolution gate:** ถ้า `plate_box` หลัง rectify กว้าง < 120 px หรือ province strip สูง < 24 px → ติดธง `low_confidence` และ**ไม่ต้อง emit จังหวัด** (ดีกว่าส่งผิดด้วยความมั่นใจ 80.83% แบบในภาพที่ 1)
3. **ตรวจ deskew ว่าไม่ทำ resolution หาย:** homography จาก 4 มุมอาจยืดภาพให้ pixel ถูก interpolate 2 รอบ (rectify → crop → resize) → ควร rectify ตรงจากภาพต้นฉบับแล้ววัดขนาดจริง
4. **นับ latency budget:** upscale + M1 บนภาพ 192px เพิ่มไม่ถึง ~5 ms แต่สำหรับภาพ 4K ต้องไม่ upscale (gate ที่ <640 เท่านั้น)

### 3.5 Post-processing (การตัดสินใจเลือกผล) — P2

1. **แก้ความไม่สอดคล้องของ validator ที่ยังเหลือ** — `has_invalid_thai_consonant_placement()` ยังห้าม `NC NNNN` (rule 4) แต่ `is_valid_plate()` ยอมรับ (`PATTERN_NC_NNNN`) → ทำให้ `5ศ 7856` ตกเข้า "Priority 0: fallback ไป CTC" แล้วถูกกู้คืนด้วย letter-plate guard ทีหลัง (ทำงานได้ แต่เปราะ: ถ้า guard ถูกแตะเมื่อไร ป้าย NC-NNNN จะกลายเป็นเลขล้วนทันที)
   **ข้อเสนอ:** ให้ `has_invalid_thai_consonant_placement()` ไม่ fire เมื่อ `is_valid_plate(text)` เป็น True (เว้นแต่มีป้ายพิเศษที่ต้องการห้ามจริง) — **ต้องตัดสินใจเชิงนโยบาย**: ถ้าผู้ใช้ยืนยันว่า `NC NNNN` (เช่น `5ศ 7856`) เป็นป้ายถูกกฎหมาย ก็ควรแก้ที่ฟังก์ชันนี้โดยตรง
2. **เพิ่มทาง "ไม่ตอบ" (abstain):**
   - `province` = "" + `review_flag` ถ้า `top1_prob < T` **หรือ** `top1 − top2 < M` (calibrate T, M บนชุดจริง E1; เริ่มที่ T=0.55, M=0.20)
   - เคสที่ควรถูกจับด้วยกฎนี้ทันที: `province_classification = 0.384` (ภาพที่ทายเป็นลาว) และกลุ่ม `ระนอง` ที่ prob ต่ำ
3. **Country gate:** ถ้า `country_confidence < 0.75` → ให้ hold/DEFAULT Thai แทน flip ไปลาว (2/481 error มาจาก flip ที่ conf 0.616) หรือรัน province head ทั้งสองภาษาแล้วเลือกจากคะแนน
4. **Prior correction สำหรับคลาสข้อมูลน้อย:** logit adjustment / temperature ตามจำนวน unique source ของคลาส — ลด prior ของคลาสที่มี <5 sources จนกว่าจะมีข้อมูลจริง (แก้ attractor `ระนอง` ได้ตรงจุดและวัดผลได้)
5. **ใช้ `alt_candidates` ให้เป็นประโยชน์จริง:** ปัจจุบันคำนวณไว้แสดงเท่านั้น → เปลี่ยนเป็น "ถ้า fused text ไม่ผ่าน `is_valid_plate` ให้ลอง candidate Top-k ที่ผ่าน validator และตรงกับ box evidence ที่สุด" (ตรงกับกลุ่ม 98 ป้ายที่ผิดตัวเดียว)
6. **กฎ syntactic เสริมสำหรับเลขซ้ำ:** ขยาย logic ที่ใช้กับพยัญชนะ (`2ถลล1238`) ไปยัง**ตัวเลขที่ซ้ำเกิน** ในป้ายรถบรรทุก (เช่น box ได้ `8 8` แต่ pattern NN-NNNN รับได้ 4 ตัว → ตัดตัวที่ prob ต่ำกว่า)
7. **ใช้ `dlt_truck_code` ที่มีอยู่แล้วตรวจไขว้:** ป้าย NN-NNNN + ชุดรหัส 70–99 = รถบรรทุก → ใช้ยืนยันจังหวัด/กันไม่ให้ refiner ทำงานผิดทาง

### 3.6 Size / scale — P2 (นิยามสัญญาให้ชัด)

- **ประกาศ input spec ขั้นต่ำ:** แนะนำ **≥ 640×360** สำหรับ production (ต่ำกว่านั้น = ติดธง)
- **งบ pixel ต่อตัวอักษร:** หลัง rectify ต้องการ **≥ 20 px สูง** ต่อตัวอักษร และ **≥ 8 px/ตัวอักษร** สำหรับ province strip — ต่ำกว่านี้ให้ low-confidence (ปัจจุบัน char crop 64×64 จากต้นทางที่สูง 12–20 px = upscale 3–5×)
- **สร้างกราฟ accuracy vs input width** (192 / 320 / 640 / 1280) บนชุดจริงชุดเดียว → ใช้ตัดสิน gate และอธิบายลูกค้าว่าภาพแบบไหนใช้ได้

---

## 4. ลำดับการทดลองที่แนะนำ (ถูกที่สุดก่อน)

| # | การทดลอง | ทำไมก่อน | ผลลัพธ์ที่ต้องการ | เกณฑ์ตัดสิน |
| :--- | :--- | :--- | :--- | :--- |
| **E1** | **Resolution-matched real-photo benchmark** — ใช้ภาพจริงของผู้ใช้ 30–50 คัน + GT (จังหวัด + ข้อความป้าย) ที่คนยืนยัน → วัด: R18 vs R34, autocontrast A/B, upscale A/B, abstain threshold | ตัวเลขที่มีอยู่ (91.89%) วัดบน 192×99 ทั้งหมด → ตัดสินใจ config จากตัวเลขนี้ไม่ได้ | ตัวเลข production-resolution + ค่า T/M ที่ calibrate ได้ | เลือก R18/R34 จากชุดนี้เท่านั้น; ตัดสินใจ autocontrast จากชุดนี้ด้วย |
| **E2** | **Re-split source-disjoint + วัดซ้ำ in-dataset** | ทำให้ "100%" เป็นตัวเลขที่เชื่อได้ (ตอนนี้ 81%/96% leak) | baseline ที่ซื่อสัตย์ของ char/province | ถ้าตัวเลขตกจาก 100% → ยืนยันข้อวินิจฉัย #2 |
| **E3** | **Harvest ภาพ full-res ~1,000 ภาพ → retrain province v6** (R18, + unknown class, + resolution aug, + prior correction) | แก้ P0 ตรงจุด | unique source ≥30/คลาส, error attractor → 0 | province บน E1 ≥95%, abstain <2% |
| **E4** | **Char v6**: เพิ่ม `junk` class + resolution/blur aug → fine-tune จาก v5 (LR ต่ำ) | แก้ `7→ษ`/`1→ข` และ guard ป้ายซ้ำ | char Top-1 บน E1 สูงกว่า v5 และ rate ของป้ายที่ผิดตัวเดียวลดลง | ป้าย exact match บน E1 เพิ่ม ≥10 จุดจากเดิม |
| **E5** | **Post-process v6**: abstain + country gate + alt_candidates + NC-NNNN policy → วัดบน E1 + 481 | ถูกและไม่ต้องเทรน | ลด error ผิดจังหวัด/ปลอม โดยไม่ลด recall | ไม่มี false "จังหวัดมั่นใจสูงแต่ผิด" เพิ่ม |

---

## 5. ตัวชี้วัดเป้าหมาย (เอาไว้ตรวจว่า v6 สำเร็จ)

| ตัวชี้วัด | ตอนนี้ | เป้า |
| :--- | :--- | :--- |
| Province unique sources ต่อคลาส (ต่ำสุด) | **1** | ≥ 30 |
| Province accuracy บนชุด real-photo (E1) | ยังไม่มี (91.89% เป็น 192px) | ≥ 95% |
| M1 no-detect rate | 2.9% (14/481, ภาพเล็ก) | ≤ 0.5% หลัง upscale gate |
| Attractor error (ทายคลาสที่มี source <5) | 12/481 | 0 |
| ป้ายผิดแบบ "ผิดตัวเดียว" | 98/374 (54% ของป้ายผิด) | ≤ 40% |
| Char unique source ต่อคลาส (ต่ำสุด) | ไม่ทราบ (ต้องนับ; crop ต่ำสุด 52) | ≥ 25 |
| GT quality ของชุดวัด | pseudo-label (`full_ocr_text`) | human-verified 200–300 ป้าย |

---

## 6. ประเด็นที่ต้องตัดสินใจ (ต้องการคำยืนยันจากเจ้าของระบบ)

1. **`NC NNNN` (เช่น `5ศ 7856`, `8ว 7687`) เป็นป้ายถูกกฎหมายหรือไม่?** — ถ้าใช่ ต้องแก้ `has_invalid_thai_consonant_placement()` rule 4 ให้สอดคล้องกับ `is_valid_plate()`; ถ้าไม่ใช่ ต้องเพิ่ม guard ไม่ให้ป้ายที่ box เห็นพยัญชนะถูกกู้คืน (ปัจจุบัน code ตั้งสมมติฐานว่า "ใช่" เพราะมี letter-plate guard)
2. **ยอมให้ระบบ "ไม่ตอบ" ได้หรือไม่** (abstain + review flag) — ถ้า workflow ปลายทางต้องการคำตอบเสมอ ต้องเปลี่ยนไปใช้ prior correction แทน (ให้คะแนนหลบคลาสข้อมูลน้อย แต่ยังตอบ)
3. **CLAHE แทน autocontrast ยอมรับได้ไหม** — ถ้าเลือก ต้องเทรนใหม่ทั้งชุดให้ contract ตรงกัน (มีค่าใช้จ่ายการเทรน)
4. **มีงบเก็บภาพจริงเพิ่ม ~1,000 ภาพสำหรับจังหวัดที่ขาดไหม** — ถ้าไม่มี ต้องยอมรับเพดานที่ต่ำลงและใช้ abstain แทน

---

*หมายเหตุการอ้างอิง:* `scratch/` เป็น gitignored — สคริปต์วัดทั้งหมด (`eval_v5i_e2e.py`, `eval_v5i_model3_plate_accuracy.py`, `eval_v4_all.py`, `eval_m3a_v2.py`) และ CSV ผลลัพธ์อยู่ในเครื่องนี้ ไม่ได้ถูก commit; ตัวเลขในเอกสารนี้คำนวณซ้ำจาก CSV เหล่านั้นและจากไฟล์ใน `datasets/Thai/`
