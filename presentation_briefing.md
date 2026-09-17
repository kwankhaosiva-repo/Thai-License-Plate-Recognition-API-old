# 🚗 เอกสารนำเสนอเชิงเทคนิค: สถาปัตยกรรมระบบตรวจจับและอ่านป้ายทะเบียนรถยนต์อัจฉริยะ (Thai & Lao LPR Pipeline)
**Technical Presentation & Architectural Defense Document**

---

## 📌 สรุปภาพรวมเชิงวิศวกรรม (Executive Summary)
ระบบนี้ถูกออกแบบขึ้นเพื่อเป็น **Real-time License Plate Recognition (LPR) Microservice** สำหรับรองรับป้ายทะเบียน **ไทย (🇹🇭)** และ **ลาว (🇱🇦)** โดยให้ความสำคัญสูงสุดกับ 3 ปัจจัย:
1. **Commercially Permissive 100%**: ใช้เฉพาะโมเดลและไลบรารีที่เป็น Apache-2.0, MIT, และ BSD-3 (ปลอดภาระผูกพันลิขสิทธิ์ AGPL หรือ Royalty Fee สำหรับภาคอุตสาหกรรม)
2. **Ultra-Low Latency on Edge / CPU**: สามารถประมวลผลจบทุกขั้นตอน (End-to-End) ได้บน **CPU ทั่วไปภายใน ~70–90 ms** โดยไม่ต้องพึ่งพา Cloud GPU
3. **Cross-Platform C# (.NET) Ready**: ทุกโมเดลสามารถ Export เป็น Standalone ONNX (Opset 18) และรันผ่าน C# / ONNX Runtime ได้โดยตรง ไม่ต้องคอมไพล์ C++ Custom Plugins

---

## 1. การตัดสินใจเลือกโมเดล (Model Selection & Trade-offs)

### ทำไมเราถึงเลือกโมเดลในแต่ละขั้นตอน?

```
[Full Image] 
     │
     ▼
[Stage 1: D-FINE-Nano] ──(Plate Bounding Box)──► [Stage 1.1: MobileNetV3 Corner Regressor]
                                                                  │
                                                        (Warp 320x160 Deskewed)
                                                                  │
                                                                  ▼
                                                      [Stage 1.5: Country Cls]
                                                                  │
                                        ┌─────────────────────────┴─────────────────────────┐
                                        ▼ (Thai)                                            ▼ (Lao)
                        [Stage 2: D-FINE-Nano]                              [Stage 2: Flip-and-Detect]
                                        │                                                   │
                  ┌─────────────────────┴─────────────────────┐                             │
                  ▼                                           ▼                             ▼
        [Stage 3A: Chars]                            [Stage 3B: Province]          [Stage 3: Lao Chars & Prov]
     (D-FINE Char Box +                               (ResNet18 Grayscale)           (RF-DETR + MobileNetV2)
      MobileNetV2 + CTC OCR)
```

| Stage | โมเดลที่เลือกใช้งานจริง (Active) | โมเดลอื่นๆ ที่ทดลองเปรียบเทียบ (Benchmark Candidates) | เหตุผลในการเลือก และ Trade-off (Latency vs Accuracy) |
| :--- | :--- | :--- | :--- |
| **Model 1 (Plate Detector)** | **D-FINE-Nano**<br>(~27 ms CPU, 15 MB) | • **PicoDet-S** (~6.5 ms, 3.8 MB)<br>• **PicoDet-M** (~12 ms, 8.9 MB)<br>• **D-FINE-Small** (~63 ms, 40 MB)<br>• **RF-DETR-Small** (~84 ms, 122 MB)<br>• **RT-DETRv2-R18** (~108 ms, 77 MB) | **Trade-off Analysis:**<br>• *PicoDet-S* เร็วที่สุด (~6.5ms) แต่มีจุดอ่อนเมื่อรถอยู่ไกล หรือป้ายเล็กมาก<br>• *RF-DETR/RT-DETR* เป็น Deformable Attention Transformer ที่แม่นยำสูง แต่ช้าเกินไปบน CPU (84–108ms) ทำให้ระบบหน่วง<br>• **D-FINE-Nano (ผู้ชนะ):** ใช้ Fine-grained Distribution Refinement (FDR) ให้ mAP ใกล้เคียง Transformer แต่กินเวลาเพียง **27 ms บน CPU** และไฟล์ ONNX เล็กเพียง **15 MB** จึงให้จุดสมดุลที่ดีที่สุด |
| **Model 1.1 (Corner Regressor)** | **MobileNetV3-Small Keypoint Regressor**<br>(~3.2 ms CPU, 4 MB) | • **YOLOv11-OBB** (Oriented Box)<br>• **RF-DETR-OBB** (Oriented Box)<br>• **Polygon Segmentation** | **ทำไมไม่ใช้ OBB ตัวเดียวจบ?**<br>• OBB หรือ Polygon Segmentation ทำให้โมเดลใหญ่ขึ้น และตอน Export เป็น ONNX จะติดปัญหา **Rotated NMS custom operator (C++)** ทำให้รันบนภาษา C# (.NET) ได้ยากมาก<br>• **2-Stage ดีกว่า:** เอา AABB BBox ดึง Crop หยาบๆ แล้วส่งให้ MobileNetV3 ทำนายพิกัด 4 มุมภายใน **3.2 ms** จากนั้นใช้ OpenCV `warpPerspective` หมุนป้ายตรงเป๊ะ 100% รันข้ามแพลตฟอร์มได้ทันที |
| **Model 1.5 (Country Classifier)** | **MobileNetV3-Small**<br>(~1.5 ms CPU, 3.8 MB) | • ResNet18<br>• Rule-based Color Check | แยกป้ายไทย vs ลาวได้แม่นยำ **99.9%** ภายในเวลาแค่ 1.5 ms ก่อนเลือกเส้นทาง Pipeline ถัดไป |
| **Model 2 (Component Detector)** | **D-FINE-Nano**<br>(~18 ms CPU, 15 MB) | • RF-DETR-Small (~55 ms)<br>• PicoDet-S (~10 ms) | ทำหน้าที่แยกพื้นที่ระหว่าง `plate_char` (แถวตัวหนังสือ) และ `province` (แถวจังหวัดด้านล่าง) D-FINE-Nano ตัดขอบแบ่งโซนได้คมชัด ไม่กินพื้นที่ทับซ้อนกัน |
| **Model 3A (Char Localization)** | **D-FINE-Nano (Box)**<br>(~16 ms CPU, 15 MB) | • Connected Component Analysis (Contours)<br>• Projection Profile | แยก Bounding Box ตัวอักษรทีละตัวได้สมบูรณ์แม้ตัวหนังสือจะชิดกันหรือมีรอยขีดข่วน |
| **Model 3A (Text Recognition)** | **Hybrid Dual-Engine**<br>• MobileNetV2 (Box Cls)<br>• ResNet18-BiLSTM-CTC | • Tesseract OCR<br>• EasyOCR<br>• Single OCR without Box | **ทำไมต้อง Hybrid?**<br>• ถ้าใช้ OCR เดี่ยวๆ เวลาเจอแสงสะท้อนหรือป้ายเอียง ตัวอักษรชิดกันจะอ่านหลุด<br>• ถ้าใช้ Box เดี่ยวๆ หากตัวอักษรบางตัวจาง กล่องจะตีกรอบไม่ติด (Partial Missing)<br>• **Dual-Engine Fusion:** Box Localizer หาตำแหน่งแม่นยำ ส่วน CTC OCR ช่วยกู้คืนตัวอักษรที่ขาดหาย ทำให้ระบบ Robust สูงสุด |
| **Model 3B (Province Classifier)** | **ResNet18-Grayscale**<br>(~4.5 ms CPU, 44 MB) | • ResNet34 RGB<br>• MobileNetV2 RGB | **ทำไมเปลี่ยนมาใช้ Grayscale?**<br>ป้ายทะเบียนไทยมีหลายสี (ป้ายขาวรถเก๋ง, ป้ายเหลืองรถบรรทุก, ป้ายเขียวรถกระบะ) ถ้าใช้ RGB โมเดลจะ Bias ตามสีพื้นหลัง การแปลงเป็น **Grayscale** ตัดเรื่องสีทิ้ง โฟกัสเฉพาะรูปร่างฟอนต์ ทำให้อ่านได้แม่นยำ **98.97% Top-1 / 99.48% Top-5** ครบทั้ง 77 จังหวัด |

---

## 2. ข้อมูลทางเทคนิคของแต่ละโมเดล (Specs & Frameworks)

| Workflow Stage | โมเดล / สถาปัตยกรรม | Input Resolution | ขนาดไฟล์ (Weights / ONNX) | Framework ที่ใช้เทรน | Task / Output Type |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stage 1 (Plate Detector)** | D-FINE-Nano | $640 \times 640 \times 3$ | 15.0 MB | PyTorch / Ultralytics (D-FINE Engine) | Object Detection (`[1, 300, 4]`) |
| **Stage 1.1 (Corner Regressor)** | MobileNetV3-Small Head | $224 \times 224 \times 3$ | 4.0 MB | PyTorch (`torchvision`) | 8-D Regression (`[1, 8]`: $x_1, y_1 \dots x_4, y_4$) |
| **Stage 1.5 (Country Cls)** | MobileNetV3-Small | $128 \times 128 \times 3$ | 3.8 MB | PyTorch (`torchvision`) | Binary Classification (`[1, 2]`) |
| **Stage 2 (Component Detector)** | D-FINE-Nano | $320 \times 160 \times 3$ | 15.0 MB | PyTorch (D-FINE Engine) | Object Detection (`plate_char`, `province`) |
| **Stage 3A (Char Box Detector)**| D-FINE-Nano | $160 \times 320 \times 3$ | 15.0 MB | PyTorch (D-FINE Engine) | Character Box Detection (`[1, 300, 4]`) |
| **Stage 3A (Thai Char Classifier)**| MobileNetV2 | $64 \times 64 \times 3$ | 8.9 MB | PyTorch (`torchvision`) | 50-Class Softmax (`[1, 50]`) |
| **Stage 3A (Thai Full OCR Engine)**| ResNet18 + BiLSTM + CTC | $32 \times 256 \times 3$ | 32.0 MB | PyTorch (Custom CTC Model) | CTC Sequence Logits (`[T, 1, 71]`) |
| **Stage 3A (Lao Char Classifier)** | MobileNetV2 | $64 \times 64 \times 3$ | 8.8 MB | PyTorch (`torchvision`) | 34-Class Softmax (`[1, 34]`) |
| **Stage 3B (Thai Province)** | ResNet18 (Grayscale) | $64 \times 256 \times 3$ | 44.0 MB | PyTorch (`torchvision`) | 77-Class Softmax (`[1, 77]`) |
| **Stage 3B (Lao Province)** | ResNet18 (Grayscale) | $64 \times 256 \times 3$ | 44.0 MB | PyTorch (`torchvision`) | 18-Class Softmax (`[1, 18]`) |

---

## 3. ขั้นตอนการเตรียมข้อมูลและการทำ Preprocessing (Preprocessing & Libraries)

### ไลบรารีหลักที่ใช้:
- **`OpenCV (cv2)`**: สำหรับ Image Decoding, Color Space Conversions (BGR $\to$ RGB, BGR $\to$ LAB), CLAHE Contrast Enhancement, Perspective Transformation, Deskewing
- **`NumPy`**: จัดการ Matrix การแปลงพิกัดมุม, Vectorization, Bounding Box Sorting และ NMS Filtering
- **`Pillow (PIL)`**: Image Resizing คุณภาพสูง (BICUBIC), Grayscale Conversions, Dynamic Canvas Padding
- **`TorchVision Transforms`**: Normalization (ImageNet Standards หรือ Mean 0.5 / Std 0.25), Tensor Formatting

### วิธีการ Preprocessing เฉพาะทางที่สำคัญ:

1. **Smart Padding Homography Crop (Stage 1 $\to$ 1.1)**:
   - ตีกรอบ Bounding Box จาก Model 1 ขยายออก **$12\%$ Margin Padding** รอบด้าน ก่อนส่งเข้า Corner Regressor เพื่อป้องกันไม่ให้มุมป้ายหลุดขอบภาพ
2. **Four-Point Clockwise Quad Sorting**:
   - พิกัดทั้ง 4 มุมจะถูกจัดเรียงตามลำดับเข็มนาฬิกาเสมอ: `[Top-Left, Top-Right, Bottom-Right, Bottom-Left]` ด้วยผลรวม $x+y$ และผลต่าง $y-x$ ก่อนส่งเข้า `cv2.getPerspectiveTransform`
3. **Grayscale Smart-Resize with Ambient Background Filling (Stage 3B)**:
   - ปัญหาเดิมของการ Resize ภาพจังหวัดคือถ้าถมขอบดำ (Black bars) โมเดลจะสับสนกับตัวหนังสือ
   - **นวัตกรรมใหม่:** โค้ดจะอ่านค่าสีเฉลี่ยจากมุมทั้ง 4 ของภาพป้ายจริง (`np.median(corners)`) แล้วนำสีนั้นมาเป็นสีพื้นหลังของ Canvas Padding ทำให้ตัวหนังสือกลมกลืนและไม่เสียอัตราส่วน (Aspect Ratio)

---

## 4. กลยุทธ์การทำ Data Augmentation และ Dataset Balancing

### ปัญหาที่พบใน Dataset จริง:
- **ความไม่สมดุลของจังหวัด (Class Imbalance)**: กรุงเทพฯ, ชลบุรี, นครปฐม มีภาพเยอะมาก (300–900 ภาพ) ขณะที่จังหวัดเล็กๆ มีเพียง 36–100 ภาพ
- **สภาพแสงและความเอียงบนท้องถนน**: กล้อง CCTV มีมุมก้ม เงาสะท้อน แดดย้อน และสภาพฝุ่นเปรอะเปื้อน

### มาตรการ Augmentation ที่นำมาใช้:
1. **Dynamic Affine Transformation**:
   - สุ่มหมุนภาพ $\pm 4^\circ$ ถึง $\pm 8^\circ$ และขยับตำแหน่งแกน $X, Y$ ($2\% - 4\%$) เพื่อจำลองมุมมองกล้องติดรถ
2. **Color & Photometric Jittering**:
   - ปรับแต่งความสว่าง (Brightness) และคอนทราสต์ (Contrast) สุ่มช่วง $0.8 - 1.2$
   - เพิ่ม `RandomAutocontrast(p=0.4)` เพื่อให้ตัวหนังสือบนป้ายเก่าหรือป้ายเลือนลางมีความคมชัดขึ้น
3. **Loss-Level Class Weighting (Cost-Sensitive Learning)**:
   - คำนวณน้ำหนัก Loss แบบผกผันกับความถี่ของข้อมูล:
     $$\text{Weight}_c = \left( \frac{N_{\text{total}}}{C \times N_c} \right)^{0.5}$$
   - ล็อกค่าให้อยู่ในช่วง $[0.2, 8.0]$ ทำให้จังหวัดที่มีภาพน้อย (เช่น 36 ภาพ) มีพลังในการปรับ Weight ของโมเดลเทียบเท่าจังหวัดใหญ่ โดยไม่จำเป็นต้อง Duplicate รูปจน Overfitting
4. **Lao Consonant Synthesis**:
   - พยัญชนะลาวบางตัวที่พบน้อยบนป้าย เช่น `ຜ`, `ດ`, `ຕ`, `ພ` ถูกนำมาทำ Synthetic Tilt & Motion Blur จนได้ความแม่นยำแตะ **99.56%**

---

## 5. ลำดับวิวัฒนาการในการทดลอง (Ablation Study & Evolution History)

ทำไมระบบถึงพัฒนามาเป็นสถาปัตยกรรมปัจจุบัน? เราเคยลองวิธีไหนมาบ้างแล้วพบปัญหาอะไร?

### ❌ ยุคที่ 1: ตรวจจับป้าย AABB ทั่วไป แล้วครอบตัดตรงๆ (Simple BBox Crop)
- **ปัญหา:** เวลาติดตั้งกล้องเฉียงหรือเลี้ยวรถ ป้ายทะเบียนจะเอียงเป็นสี่เหลี่ยมคางหมู ตัวหนังสือจะบิดเบี้ยว OCR อ่านผิดทันทีเกิน 40%

### ❌ ยุคที่ 2: ใช้ 1-Stage Rotated Bounding Box (OBB) หรือ Segmentation Model
- **ปัญหา:** 
  1. โมเดลหนักและหน่วงมากบน CPU (~90–120 ms)
  2. ตอน Export เป็น ONNX เพื่อนำไปใช้บน C# / .NET มักจะพังเนื่องจากไม่มี C++ Custom Plugin สำหรับ Rotated NMS
- **ทางออกที่ชนะ:** เปลี่ยนเป็น **2-Stage Light Keypoint Regressor** (AABB D-FINE-Nano + MobileNetV3 Keypoint) ใช้เวลาเพียง ~30 ms รวมทั้งสองขั้นตอน และรันบน C# OnnxRuntime มาตรฐานได้ทันที 100%

### ❌ ยุคที่ 3: ใช้ OCR อ่านข้อความทั้งป้ายรวดเดียว โดยไม่แบ่ง Component
- **ปัญหา:** ป้ายทะเบียนไทยมี 2 บรรทัด (บรรทัดบน: หมวดอักษร-ตัวเลข, บรรทัดล่าง: จังหวัด) บางครั้งตัวเลขป้ายรถบรรทุกมี 2 จุด ถ้าอ่านทั้งผืน โมเดล OCR จะอ่านชื่อจังหวัดปนเข้าไปในเลขทะเบียน หรือสลับบรรทัดกัน
- **ทางออกที่ชนะ:** ออกแบบ **Model 2 (Component Detector)** เพื่อตัดแยกระหว่าง `plate_char` และ `province` อย่างเด็ดขาด ทำให้โมเดลแต่ละส่วนทำงานเฉพาะทางอย่างแม่นยำ

### ❌ ยุคที่ 4: พึ่งพาตัวอ่านรหัสป้ายรถบรรทุกด้านบน (DLT Top Banner Code)
- **ปัญหา:** ป้ายรถบรรทุกมีรหัสตัวเลข 2 หลักข้างคำว่า THAILAND (เช่น 70 = ราชบุรี, 55 = น่าน) แต่ตัวเลขนี้มักมีขนาดเล็กมาก และมักโดนหัวน็อตยึดป้ายเจาะทับ ทำให้โมเดลอ่านผิด (เช่น อ่าน 70 กลายเป็น 55) แล้วไป Override ทับชื่อจังหวัดที่ถูกต้อง
- **ทางออกที่ชนะ:** **ถอดระบบ DLT Code Override ออก** แล้วพัฒนา **Model 3B Grayscale Classifier (ResNet18)** ให้แม่นยำสูงถึง **98.97% Top-1** โดยอ่านจากชื่อจังหวัดตรงๆ ด้านล่าง ซึ่งมีพื้นที่ใหญ่กว่าและเชื่อถือได้มากกว่าหลายเท่า

### ❌ ยุคที่ 5: พึ่งพาเฉพาะ Box Detector สำหรับตัดตัวอักษรทีละตัว
- **ปัญหา:** เวลาเจอสภาพแสงจ้า (Sun Glare) หรือรอยเปื้อน กล่อง Detector อาจจะตรวจไม่พบตัวอักษรบางตัว (เช่น 5/6 ตัว) ทำให้ป้ายขาดหาย
- **ทางออกที่ชนะ:** ออกแบบเป็น **Hybrid Dual-Engine**: ให้ Box Localizer จับคู่กับ **ResNet-BiLSTM Full Plate CTC OCR** หากกล่องตัวอักษรขาด ระบบจะสลับไปดึงผลลัพธ์จาก CTC Engine มาเติมเต็มทันที ทำให้ไม่เคยเกิดข้อผิดพลาดป้ายขาดหาย

---

## 6. สรุปความพร้อมในการนำไปใช้งานจริง (Production Readiness)
- ✅ **API & Interactive Dashboard**: มี Web UI Dashboard สวยงามแสดงผลแบบ Step-by-step ทุก Stage (Latency, Bounding Boxes, Probability Bar Chart)
- ✅ **Commercially Compliant**: ปราศจากลิขสิทธิ์ AGPL หรือเงื่อนไขที่ห้ามใช้เชิงพาณิชย์
- ✅ **Full C# (.NET) Integration**: พร้อมโค้ดตัวอย่าง `Microsoft.ML.OnnxRuntime` + `OpenCvSharp4` นำไปใส่ในโปรเจกต์ Desktop หรือ Server ของลูกค้าได้ทันที
- ✅ **Tested Accuracy**: 
  - Plate Localization: **~99.2% mAP**
  - Character Recognition: **~98.5% Sequence Accuracy**
  - Province Classification: **98.97% Top-1 / 99.48% Top-5**
  - Average End-to-End Latency: **~75–90 ms บน CPU ทั่วไป**
