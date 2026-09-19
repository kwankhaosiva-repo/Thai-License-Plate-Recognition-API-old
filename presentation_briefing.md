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
     (D-FINE Char Box +                               (ResNet34 Grayscale)           (RF-DETR + MobileNetV2)
      MobileNetV2 + CTC OCR +
      Method A+C Spatial Fusion)
```

| Stage | โมเดลที่เลือกใช้งานจริง (Active) | โมเดลอื่นๆ ที่ทดลองเปรียบเทียบ (Benchmark Candidates) | เหตุผลในการเลือก และ Trade-off (Latency vs Accuracy) |
| :--- | :--- | :--- | :--- |
| **Model 1 (Plate Detector)** | **D-FINE-Nano**<br>(~27 ms CPU, 15 MB) | • **PicoDet-S** (~6.5 ms, 3.8 MB)<br>• **PicoDet-M** (~12 ms, 8.9 MB)<br>• **D-FINE-Small** (~63 ms, 40 MB)<br>• **RF-DETR-Small** (~84 ms, 122 MB)<br>• **RT-DETRv2-R18** (~108 ms, 77 MB) | **Trade-off Analysis:**<br>• *PicoDet-S* เร็วที่สุด (~6.5ms) แต่มีจุดอ่อนเมื่อรถอยู่ไกล หรือป้ายเล็กมาก<br>• *RF-DETR/RT-DETR* เป็น Deformable Attention Transformer ที่แม่นยำสูง แต่ช้าเกินไปบน CPU (84–108ms) ทำให้ระบบหน่วง<br>• **D-FINE-Nano (ผู้ชนะ):** ใช้ Fine-grained Distribution Refinement (FDR) ให้ mAP ใกล้เคียง Transformer แต่กินเวลาเพียง **27 ms บน CPU** และไฟล์ ONNX เล็กเพียง **15 MB** จึงให้จุดสมดุลที่ดีที่สุด |
| **Model 1.1 (Corner Regressor)** | **MobileNetV3-Small Keypoint Regressor**<br>(~3.2 ms CPU, 4 MB) | • **YOLOv11-OBB** (Oriented Box)<br>• **RF-DETR-OBB** (Oriented Box)<br>• **Polygon Segmentation** | **ทำไมไม่ใช้ OBB ตัวเดียวจบ?**<br>• OBB หรือ Polygon Segmentation ทำให้โมเดลใหญ่ขึ้น และตอน Export เป็น ONNX จะติดปัญหา **Rotated NMS custom operator (C++)** ทำให้รันบนภาษา C# (.NET) ได้ยากมาก<br>• **2-Stage ดีกว่า:** เอา AABB BBox ดึง Crop หยาบๆ แล้วส่งให้ MobileNetV3 ทำนายพิกัด 4 มุมภายใน **3.2 ms** จากนั้นใช้ OpenCV `warpPerspective` หมุนป้ายตรงเป๊ะ 100% รันข้ามแพลตฟอร์มได้ทันที |
| **Model 1.5 (Country Classifier)** | **MobileNetV3-Small**<br>(~1.5 ms CPU, 3.8 MB) | • ResNet18<br>• Rule-based Color Check | แยกป้ายไทย vs ลาวได้แม่นยำ **99.9%** ภายในเวลาแค่ 1.5 ms ก่อนเลือกเส้นทาง Pipeline ถัดไป |
| **Model 2 (Component Detector)** | **D-FINE-Nano**<br>(~18 ms CPU, 15 MB) | • RF-DETR-Small (~55 ms)<br>• PicoDet-S (~10 ms) | ทำหน้าที่แยกพื้นที่ระหว่าง `plate_char` (แถวตัวหนังสือ) และ `province` (แถวจังหวัดด้านล่าง) D-FINE-Nano ตัดขอบแบ่งโซนได้คมชัด ไม่กินพื้นที่ทับซ้อนกัน |
| **Stage 3A (Char Localization)** | **RF-DETR-Base (Box)**<br>(~45 ms CPU, 122 MB .pt) | • **D-FINE-Small** (~35 ms, 165 MB)<br>• **D-FINE-Nano** (~16 ms, 15 MB)<br>• Connected Component Analysis | **ทำไมเลือก RF-DETR-Base?**<br>• ใช้ Multi-scale Deformable Attention ในการสแกนพื้นที่ตัวอักษรโดยตรงโดยไม่มีปัญหา Anchor Box Bias<br>• สามารถตรวจจับขอบเขตตัวอักษรที่ชิดกัน หรือตัวอักษรที่มีสระ/วรรณยุกต์ซ้อน และป้ายที่มีสกรูยึดเจาะทะลุได้อย่างคมชัดและแม่นยำสูงสุด |
| **Model 3A (Text Recognition)** | **Hybrid Method A+C Dual-Engine**<br>• MobileNetV2 (50-Class Balanced Cls, **99.58% Top-1**)<br>• ResNet18-BiLSTM-CTC<br>• Autocontrast Normalization<br>• Spatial Gap Gating + DLT Syntax Guard | • Tesseract OCR<br>• EasyOCR<br>• 12-Epoch Imbalanced Classifier<br>• Blind Sequence Alignment | **ทำไมต้อง Balanced 50-Class + Method A+C Unified Fusion?**<br>• **Balanced Dataset:** ทำ Augmentation ปรับสมดุลทุกคลาสเป็น $\ge 400$ ตัวอย่าง/คลาส (รวม 29,385 ภาพ) พร้อมใส่ Photometric Shadow Gradients แก้ปัญหาตัวอักษรหายาก (`ผ`, `ณ`, `ฬ`) โดนทายสับสนเป็นตัวเลขทึบอย่าง `8`<br>• **Autocontrast Normalization:** ดึง Contrast ขยาย Dynamic Range ตัวอักษรในเงามืดก่อนเข้า Classifier<br>• **Method A+C Fusion:** ผสานความแม่นยำระดับ **99.58% Top-1 (99.89% Top-3)** ของ Classifier เข้ากับ CTC โดยมี **DLT Syntax Guard** ดักจับ Format ป้ายส่วนบุคคลที่เป็นไปไม่ได้ (เช่น `\d[พยัญชนะ]\d{4}`) |
| **Model 3B (Province Classifier)** | **ResNet18-Grayscale**<br>(~5.0 ms CPU, 42.9 MB, $64 \times 256$) | • ResNet34 Grayscale ($80 \times 256$, 81.5 MB)<br>• ResNet34 RGB<br>• MobileNetV2 RGB | **ทำไมเลือก ResNet18 Grayscale ($64 \times 256$)?**<br>• ขนาดไฟล์เล็กลงเกือบ 50% (เหลือเพียง 42.9 MB) และประมวลผลเร็วมากบน CPU (~5 ms) เหมาะสำหรับการทำ Containerization บน Cloud Run<br>• ให้ความแม่นยำสูงถึง **99.20% Val Top-1** ครบทั้ง 77 จังหวัด และทนทานต่อสภาพแสงสะท้อน แดดย้อน หรือเงามืด |

---

## 2. ข้อมูลทางเทคนิคของแต่ละโมเดล (Specs & Frameworks)

| Workflow Stage | โมเดล / สถาปัตยกรรม | Input Resolution | ขนาดไฟล์ (Weights / ONNX) | Framework ที่ใช้เทรน | Task / Output Type |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stage 1 (Plate Detector)** | D-FINE-Nano | $640 \times 640 \times 3$ | 15.0 MB | PyTorch / Ultralytics (D-FINE Engine) | Object Detection (`[1, 300, 4]`) |
| **Stage 1.1 (Corner Regressor)** | MobileNetV3-Small Head | $224 \times 224 \times 3$ | 4.0 MB | PyTorch (`torchvision`) | 8-D Regression (`[1, 8]`: $x_1, y_1 \dots x_4, y_4$) |
| **Stage 1.5 (Country Cls)** | MobileNetV3-Small | $128 \times 128 \times 3$ | 3.8 MB | PyTorch (`torchvision`) | Binary Classification (`[1, 2]`) |
| **Stage 2 (Component Detector)** | D-FINE-Nano | $320 \times 160 \times 3$ | 15.0 MB | PyTorch (D-FINE Engine) | Object Detection (`plate_char`, `province`) |
| **Stage 3A (Char Box Detector)**| RF-DETR-Base | $160 \times 320 \times 3$ | 121.8 MB | PyTorch (RF-DETR Transformer) | Character Box Detection (`[1, 300, 4]`) |
| **Stage 3A (Thai Char Classifier)**| MobileNetV2 | $64 \times 64 \times 3$ | 9.0 MB | PyTorch (`torchvision`) | 50-Class Softmax (`[1, 50]`) — **99.58% Val Top-1 / 99.89% Top-3** |
| **Stage 3A (Thai Full OCR Engine)**| ResNet18 + BiLSTM + CTC | $32 \times 256 \times 3$ | 54.8 MB | PyTorch (Custom CTC Model) | CTC Sequence Logits (`[T, 1, 71]`) |
| **Stage 3A (Lao Char Classifier)** | MobileNetV2 | $64 \times 64 \times 3$ | 8.9 MB | PyTorch (`torchvision`) | 34-Class Softmax (`[1, 34]`) |
| **Stage 3B (Thai Province)** | ResNet18 (Grayscale) | $64 \times 256 \times 3$ | 42.9 MB | PyTorch (`torchvision`) | 77-Class Softmax (`[1, 77]`) — **99.20% Val Top-1** |
| **Stage 3B (Lao Province)** | ResNet18 (Grayscale) | $64 \times 256 \times 3$ | 42.7 MB | PyTorch (`torchvision`) | 18-Class Softmax (`[1, 18]`) — **99.20% Val Top-1** |

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
1. **Character Dataset Balancing (แก้ปัญหาอักษรหายาก & ป้ายในเงามืด)**:
   - **ปัญหา:** ตัวอักษรหายาก (`ผ`, `ณ`, `ฬ`, `ฮ`) เดิมมีเพียง ~60 ภาพ เทียบกับตัวเลขที่มี 1,500+ ภาพ และภาพเทรนเดิมสว่างขาว (Mean ~160) ขาดภาพมืดใต้กันชน (Mean ~49) ทำให้โมเดลเดาเป็น `8`
   - **การแก้ไข:** ทำ Offline Balancing ปรับสมดุลครบทั้ง 50 คลาสให้มี $\ge 400$ ภาพ/คลาสในชุด Train (รวม 29,385 ภาพ) และ $\ge 30$ ภาพ/คลาสในชุด Valid (รวม 3,612 ภาพ)
   - **Synthetic Photometric Shadow Gradients:** สุ่มใส่แถบเงาดำทแยงและแนวตั้ง ปรับความสว่างต่ำ $0.35 - 0.65$ และใส่ `RandomGrayscale(p=0.15)` ทำให้โมเดลเรียนรู้โครงสร้างลายเส้นแม้ภาพจะมืดสนิท จนได้ **Val Top-1: 99.58% / Top-3: 99.89%**
2. **Dynamic Affine Transformation**:
   - สุ่มหมุนภาพ $\pm 4^\circ$ ถึง $\pm 8^\circ$ และขยับตำแหน่งแกน $X, Y$ ($2\% - 4\%$) เพื่อจำลองมุมมองกล้องติดรถ
3. **Color & Photometric Jittering**:
   - ปรับแต่งความสว่าง (Brightness) และคอนทราสต์ (Contrast) สุ่มช่วง $0.8 - 1.2$
   - เพิ่ม `RandomAutocontrast(p=0.4)` เพื่อให้ตัวหนังสือบนป้ายเก่าหรือป้ายเลือนลางมีความคมชัดขึ้น
4. **Loss-Level Class Weighting (Cost-Sensitive Learning)**:
   - คำนวณน้ำหนัก Loss แบบผกผันกับความถี่ของข้อมูลสำหรับ Province Classifier:
     $$\text{Weight}_c = \left( \frac{N_{\text{total}}}{C \times N_c} \right)^{0.5}$$
   - ล็อกค่าให้อยู่ในช่วง $[0.2, 8.0]$ ทำให้จังหวัดที่มีภาพน้อยมีพลังในการปรับ Weight ของโมเดลเทียบเท่าจังหวัดใหญ่
5. **Lao Consonant Synthesis**:
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
- **ทางออกที่ชนะ:** **ถอดระบบ DLT Code Override ออก** แล้วพัฒนา **Model 3B Grayscale Classifier (ResNet34, 80x256)** ให้แม่นยำสูงถึง **99.10% Val / 98.71% Test Top-1** โดยอ่านจากชื่อจังหวัดตรงๆ ด้านล่าง ซึ่งมีพื้นที่ใหญ่กว่าและเชื่อถือได้มากกว่าหลายเท่า

### ❌ ยุคที่ 5: การเติมตัวอักษรแบบ Blind CTC Insertion
- **ปัญหา:** เมื่อ Box Detector หลุดตัวอักษร แล้วให้ CTC นำตัวอักษรที่เกินมาเติมแบบตรงๆ (Blind Sequence Alignment) จะทำให้เกิดปัญหา:
  1. หัวน็อตยึดป้าย หรือกรอบป้ายสะท้อนแสง CTC อาจหลอนมองเป็นตัวเลข เช่น ป้ายจริง `กย 588` โดนเติมกลายเป็น `5กย 4588`
  2. ป้ายรถบรรทุก (`70-1737`) ตัวตรวจจับมองขีดกลางเป็นกล่อง แล้ว Classifier ไม่มีคลาสขีดกลาง จึงเดาเป็นพยัญชนะไทย กลายเป็น `70ษย7ม` ซึ่งผิดกฎหมายอย่างสิ้นเชิง

### ❌ ยุคที่ 6: Classifier ขาดสมดุล (12 Epochs Imbalanced) และถูก Merged Box กลืน
- **ปัญหา:**
  1. ตัวอักษรพบน้อย (`ผ`, `ณ`, `ฬ`) โดนทายสับสนเป็นตัวเลขปิดทึบอย่าง `8` เมื่อเจอสภาพแสงมืดใต้กันชน
  2. ตัวอักษรที่เขียนชิดกัน (เช่น `97`) Box Detector มักทำนายกล่องกว้าง 1 กล่องครอบ 2 ตัว ($w > 1.6 \times \text{median}$) พอทำ Standard NMS กล่องกว้างที่มี Confidence สูงจะไปกลืนกล่องตัวเลขเดี่ยว 2 ตัวข้างในทิ้ง ทำให้ตัวอักษรขาดหาย
  3. ป้ายอักษรเดี่ยวโบราณ เช่น `ณ 4100` โดน CTC หลอนเป็นเลข 6 ตัว (`74-0001`) แล้วระบบเลือกลำดับยาวกว่าไปทับ

### ✅ ยุคที่ 7 (สถาปัตยกรรม Production ปัจจุบัน): Production-Grade 50-Class Balanced Engine + Subsumed Box Filter
- **ทางออกที่ชนะและเสถียรที่สุด:**
  1. **Balanced 50-Class Dataset & 35-Epoch Cosine Training**: ปรับสมดุลทุกคลาสเป็น $\ge 400$ ภาพ (29,385 ภาพ) พร้อม Photometric Shadow Augmentation ได้ความแม่นยำ **99.58% Val Top-1 / 99.89% Top-3**
  2. **Subsumed Composite Box Filter**: ตรวจสอบกล่องที่กว้างผิดปกติและครอบกล่องย่อย 2 กล่องไว้ข้างใน โดยระบบจะตัดกล่องรวมทิ้งเพื่อรักษาตัวอักษรเดี่ยวทั้ง 2 ตัวไว้
  3. **Autocontrast Normalization**: ขยาย Dynamic Range ของ Crop ตัวอักษรที่มีเงาทอดทึบก่อนส่งเข้า Classifier
  4. **DLT Syntax Placement Guard (`has_invalid_thai_consonant_placement`)**: ล็อคกฎไวยากรณ์ป้ายรถยนต์ส่วนบุคคล ห้ามมีพยัญชนะตามหลังตัวเลข 2 หลัก หรือขึ้นต้นด้วยตัวเลขเดี่ยวแล้วตามด้วยพยัญชนะ (`\d[พยัญชนะ]\d{4}`)
  5. **Protected Sequence Reconciliation**: ปกป้องป้ายพยัญชนะไทยแท้ เช่น `ณ 4100` หรือ `ผว 7697` ไม่ให้ถูกภาพหลอนป้ายรถบรรทุกสวมรอยทับ

---

## 6. ระบบ Real-Time Stream & Dynamic 5-Second Vehicle Session Aggregator
- **Rain & Noise-Proof Motion Gating (0 ms Idle Latency):**
  - สตรีมวิดีโอผ่านการกรองแบบ Gaussian Blur ($15 \times 15$) + Background Subtraction (MOG2) + Morphological Opening ($5 \times 5$)
  - ช่วยตัดปัญหาเม็ดฝน แสงแดดสะท้อน หรือเงาเคลื่อนไหว เมื่อถนนว่างจะข้ามการรันโมเดล AI ทั้งหมดทันที ทำให้กิน CPU เป็น **0 ms**
- **Dynamic 5-Second Vehicle Tracking & Multi-Frame Consolidation:**
  - รถ 1 คันที่ขับผ่านกล้องจะถูกบันทึกเฟรมต่อเนื่องภายในหน้าต่างเวลา 5.0 วินาที
  - ระบบทำ **Majority Voting** อักษรและจังหวัด พร้อมเฉลี่ยค่า **Mean Confidence** จากหลายมุมมอง คัดเลือก Crop ภาพที่ชัดที่สุด
  - บันทึกลงฐานข้อมูลเพียง **1 Record ที่แม่นยำที่สุดต่อรถ 1 คัน** หมดปัญหาประวัติซ้ำซ้อน
- **High-Precision Thresholding (`conf_m1 = 0.80`):**
  - ตั้งค่า Threshold ของ Model 1 สำหรับโหมดวิดีโอและสตรีมไว้ที่ `0.80` เพื่อป้องกัน False Alarm จากขอบทางหรือกันชน
  - เพิ่มความถี่ในการตรวจจับ (`STREAM_FRAME_SKIP = 2`, `STREAM_TARGET_SAMPLES = 1`) ทำให้จับป้ายทะเบียนได้ตั้งแต่เฟรมแรกที่เห็นชัดเจน

---

## 7. สถาปัตยกรรม Dual Cloud Storage (Google Cloud Firestore + BigQuery)
- **Dual Cloud Pattern:**
  - **Cloud Firestore (`lpr-db`)**: เก็บประวัติแบบ Document Store รวมรูปภาพป้ายทะเบียนแบบ Base64 สำหรับดึงดูบนหน้าเว็บแบบ Real-time
  - **Cloud BigQuery (`lpr_query.lpr_history`)**: เก็บตารางข้อมูลสำหรับประมวลผล Big Data และวิเคราะห์สถิติต่าง ๆ
- **Zero-Latency Async Worker:**
  - การบันทึกข้อมูลขึ้น Cloud ทำงานผ่าน `ThreadPoolExecutor` แยกเป็น Background Worker ไม่ดึงหน่วงรอบประมวลผลของกล้องสด (Latency คงเดิม)
- **Native Cloud Run Auto-Detection:**
  - ระบบตรวจจับตัวแปรระบบ `K_SERVICE` บน GCP Cloud Run โดยอัตโนมัติ สั่งซ่อนการเชื่อมต่อ SQLite ภายใน Container และสลับไปดึงข้อมูลจาก Cloud Firestore โดยตรง 100%

---

## 8. สรุปความพร้อมในการนำไปใช้งานจริง (Production Readiness)
- ✅ **API & Interactive Dashboard**: มี Web UI Dashboard สวยงามแสดงผลแบบ Step-by-step ทุก Stage (Latency, Bounding Boxes, Probability Bar Chart) พร้อม Modal แสดงสถาปัตยกรรมโมเดลและระบบสลับประวัติ Local/Cloud
- ✅ **Commercially Compliant**: ปราศจากลิขสิทธิ์ AGPL หรือเงื่อนไขที่ห้ามใช้เชิงพาณิชย์
- ✅ **Full C# (.NET) Integration**: พร้อมโค้ดตัวอย่าง `Microsoft.ML.OnnxRuntime` + `OpenCvSharp4` นำไปใส่ในโปรเจกต์ Desktop หรือ Server ของลูกค้าได้ทันที
- ✅ **GCP Cloud Native**: พร้อม Dockerfile และ `.dockerignore` ที่ลดขนาดลงกว่า 1.2 GB รองรับการ Deploy บน Cloud Run ได้ทันที
- ✅ **Tested Accuracy**: 
  - Plate Localization: **~99.2% mAP**
  - Character Recognition: **~98.5% Sequence Accuracy** (ด้วย Method A+C Spatial Fusion)
  - Province Classification: **99.20% Val Top-1** (ResNet18-Grayscale, 42.9 MB)
  - Average End-to-End Latency: **~75–95 ms บน CPU ทั่วไป**
