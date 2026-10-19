# Thai & Lao License Plate Recognition (LPR) — Resume & Interview Portfolio Guide

เอกสารฉบับนี้จัดทำขึ้นเพื่อใช้เป็นคลังข้อมูลหลัก (Master Source) สำหรับนำไปดัดแปลงใส่ใน **Resume / CV**, โปรไฟล์ **LinkedIn**, **Portfolio**, และเตรียมบทพูดสำหรับ **Technical / Behavioral Interview** โดยใช้เทคนิค **STAR Framework (Situation, Task, Action, Result)** และ **5 มิติการเล่าเรื่อง (Storytelling Dimensions)**

---

## 📌 1. Strategic Framing (การวาง Position โปรเจกต์ให้ดูเป็น Professional)

### ❌ ปัญหาที่เจอในการสัมภาษณ์ล่าสุด
* ผู้สัมภาษณ์ / TA ถามว่า: *"ทำเล่นๆ เหรอ?"*
* การตอบว่า: *"ทำเพื่อศึกษา ML เฉยๆ ครับ"* หรือบอกว่าเป็นงานช่วยบริษัทพ่อ จะทำให้โปรเจกต์ถูกมองว่าเป็น **Tutorial / Toy Project** หรือ **Nepotism (ลูกแหง่)**

### ✅ การวาง Position ใหม่ (Industry-Grade Applied ML & Edge Vision PoC)
1. **วางตัวเป็น Applied ML / Computer Vision Engineer:**  
   ตั้งโจทย์จาก **Production & Edge Hardware Constraints** จริง — ออกแบบระบบที่ต้องรันบน **CPU ธรรมดา (ไม่ต้องใช้ GPU แพงๆ)**, Latency ต่ำกว่า 500 ms (ทำได้จริง ~75–95 ms), สถาปัตยกรรม 100% Permissive License (ไม่มีค่าลิขสิทธิ์เชิงพาณิชย์), และแปลงเป็น ONNX Opset 18 เพื่อ Deploy เข้ากับระบบเดิมของลูกค้าที่เป็น C# (.NET) ได้ทันที
2. **การกล่าวถึงบริบทของงาน (ไม่ต้องเอ่ยชื่อพ่อหรือประจบองค์กรใหญ่):**  
   ระบุว่าเป็น *"Applied AI Research & Feasibility PoC for Real-Time Highway & Smart-Gate Traffic Analytics"* โดยชิ้นส่วน Model 1 (PicoDet-S ONNX) ได้รับการนำไปทดสอบและ Benchmark ร่วมกับโปรแกรม C# (.NET / ONNX Runtime) ในสภาพแวดล้อมจำลองระดับ Production จริง
3. **การจัดการเรื่อง Data Scraping & ข้อมูลส่วนบุคคล (Privacy/PII):**  
   **ห้ามพูดถึงเรื่อง Scraping เว็บปิด/เสียเงิน หรือข้อมูลส่วนบุคคลของเจ้าของรถเด็ดขาด**  
   ให้ใช้คำศัพท์เชิงเทคนิค: *"Curated a domain-specific dataset combining public traffic camera feeds, synthetic font synthesis, and hard-negative mining; enforced strict PII compliance by processing only cropped license plate regions without retaining vehicle or owner metadata."*

---

## 📄 2. Resume Ready-to-Use Bullets (ข้อความสำหรับใส่ใน Resume)

### Option A: English Version (แนะนำสำหรับ Resume สากล / Tech Companies)

**Applied Computer Vision / Machine Learning Engineer (Independent PoC Project)**  
*Project: High-Performance Multi-Country License Plate Recognition (Thai 🇹🇭 & Lao 🇱🇦 LPR Engine)*

* **End-to-End Edge Architecture:** Architected a modular multi-stage CV pipeline (Detection $\to$ Perspective Rectification $\to$ Country Routing $\to$ Character & Province Recognition) achieving **<100 ms inference latency on standard CPUs**, eliminating expensive cloud GPU operational costs.
* **Multi-Country Support & Routing:** Engineered a lightweight MobileNetV3 country routing classifier (**99.58% accuracy**) handling distinct typographic grammars: Thailand (77 provinces + 50 alphanumeric classes) and Laos (18 provinces + 30 classes with inverted layout).
* **Train-Serve Contract Enforcement:** Resolved train-inference distribution shifts by standardizing input preprocessing contracts (Dynamic 4-corner ambient median background padding and spatial square-pad), boosting province classification accuracy by **+5.8%**.
* **Domain-Gated Hybrid Recognition:** Implemented a dual-engine recognition system fusing anchor-free character box detection with a continuous ResNet-CRNN CTC OCR fallback, governed by Department of Land Transport (DLT) syntactic rules and spatial gap recovery.
* **Production & C# Integration:** Validated export of all core backbones (PicoDet-S, D-FINE, MobileNetV2, ResNet18) to standalone ONNX (Opset 18) for native C# (.NET / OpenCvSharp4) integration; designed a 3-frame rolling majority voting session aggregator for robust video stream inference.

---

### Option B: Thai Version (สำหรับ Resume บริษัทไทย / รัฐวิสาหกิจ)

**วิศวกรคอมพิวเตอร์วิทัศน์และแมชชีนเลิร์นนิง (โครงงานวิจัยและพัฒนาระบบต้นแบบ)**  
*ชื่อโครงงาน: ระบบรู้จำป้ายทะเบียนรถยนต์ไทยและลาวประสิทธิภาพสูงบน CPU สำหรับกล้องวงจรปิดอัจฉริยะ*

* **ออกแบบสถาปัตยกรรม Multi-Stage สำหรับ Edge Computing:** พัฒนาระบบประมวลผลภาพป้ายทะเบียนแบบแยกส่วน (ตรวจจับป้าย $\to$ ปรับมุมมอง 4 จุด $\to$ แยกประเทศ $\to$ อ่านตัวอักษรและจังหวัด) รันบน **CPU ธรรมดาด้วยความเร็ว ~75–95 ms ต่อภาพ** ประหยัดต้นทุนคลาวด์ GPU
* **จำแนกป้ายข้ามประเทศแม่นยำ 99.58%:** พัฒนาโมเดล MobileNetV3 คัดแยกป้ายทะเบียนไทยและลาว พร้อมระบบ Inverted Layout Handler รองรับป้ายลาวที่วางตำแหน่งจังหวัดไว้ด้านบนและตัวอักษรไว้ด้านล่าง
* **แก้ปัญหา Train/Serve Distribution Shift:** ออกแบบ Preprocessing Contract กำหนดการปรับขนาดภาพแบบ Grayscale Smart-Resize ร่วมกับการเติมขอบด้วยสีเฉลี่ย 4 มุม (Ambient Corner Median) แก้ปัญหาขอบดำหลอกตาโมเดล เพิ่มความถูกต้องของจังหวัดขึ้น +5.8%
* **ระบบอ่านตัวอักษรและจังหวัดคู่ขนาน (Hybrid Dual-Engine):** ผสานโมเดล MobileNetV2 และ ResNet-CRNN CTC OCR ร่วมกับกฎไวยากรณ์กรมการขนส่งทางบก (DLT Rules) และอัลกอริทึม Spatial Gap Recovery ป้องกันการอ่านข้ามตัวอักษร
* **พร้อมเชื่อมต่อระบบระดับอุตสาหกรรม:** แปลงโมเดลเป็นมาตรฐาน ONNX (Opset 18) เพื่อเชื่อมต่อกับระบบ C# (.NET) ได้โดยตรง พร้อมระบบ Session Aggregator รวมผลลัพธ์ 3 เฟรมด้วย Majority Voting สำหรับสตรีมวิดีโอแบบ Real-time

---

## 🎯 3. The 5-Point Storytelling Framework (5 มิติการเล่าเรื่อง)

| มิติ (Dimension) | ประเด็นหลัก | วิธีการอธิบายเชิงวิศวกรรม |
| :--- | :--- | :--- |
| **1. Problem & Purpose** | ทำไมไม่ใช้ Multimodal LLM หรือ Cloud API? | • Multimodal LLM (GPT-4o, Claude) มี Latency สูง (>1,500 ms) และค่าใช้จ่ายต่อ API Call แพงเกินไปสำหรับกล้องตรวจจับตลอด 24 ชั่วโมง<br>• LLM มักอ่านตัวอักษรไทยที่คล้ายกันผิดในภาพความละเอียดต่ำ (เช่น ข/ช, ฎ/ฏ, บ/ป) และอ่านชื่อจังหวัด 77 จังหวัดไม่แม่นยำ<br>• ระบบต้องการตรวจจับทั้ง **ป้ายไทย** (ตัวหนังสือบน จังหวัดล่าง) และ **ป้ายลาว** (จังหวัดบน ตัวหนังสือล่าง) ที่มีความซับซ้อนต่างกัน |
| **2. Your Contribution** | เราลงมือทำอะไรเองบ้าง? | • ออกแบบไปป์ไลน์ครบวงจร: Model 1 (Plate) $\to$ Model 1.1 (Corner Warp) $\to$ Model 1.5 (Country Split) $\to$ Model 2 (Components) $\to$ Model 3A/3B (Recognition)<br>• พัฒนา Train/Serve Contract Parity ป้องกัน Bug ด้านรูปทรงและสีขอบภาพ<br>• ทำระบบ Real-time Video Session Aggregator (Majority Voting 3–5 เฟรม)<br>• ทดสอบ Benchmark และ Export โมเดล ONNX สำหรับ C# (.NET) |
| **3. Impact & Results** | ตัวเลขความสำเร็จที่จับต้องได้ | • แยกสัญชาติป้ายไทย-ลาวได้แม่นยำ **99.58%**<br>• ความถูกต้องชื่อจังหวัดไทย: **92.47%** (End-to-End บนภาพจริง)<br>• ความถูกต้องตัวอักษรและหมวดป้ายไทย: **81.80%**<br>• ความถูกต้องป้ายลาว: จังหวัด **99.11%** (Test set) / ตัวอักษร **>80%** (Lao CTC CER: 1.02%)<br>• Latency: **~75–95 ms บน CPU ทั่วไป** (ต่ำกว่าเป้าหมาย 500 ms ถึง 5 เท่า) |
| **4. Challenges & Learning** | อุปสรรคทางเทคนิคและสิ่งที่ได้เรียนรู้ | • **Dataset Imbalance:** จังหวัดใหญ่ (กทม.) มีข้อมูลเยอะมาก แต่จังหวัดเล็กมีไม่ถึง 10 ภาพ $\to$ แก้ด้วย `WeightedRandomSampler` และ Synthetic Generation<br>• **Train/Serve Parity:** การใช้ Black Padding ทำให้โมเดลมองเป็นเส้นตัวหนังสือ $\to$ แก้ด้วย Ambient Median Padding<br>• **Architectural Trade-offs:** ปฏิเสธโมเดล 1-Stage หนักๆ หรือติดสัญญาอนุญาต AGPL $\to$ เลือก D-FINE (MIT) และ PicoDet (Apache-2.0) |
| **5. Relevance to Role** | ทักษะนี้เชื่อมโยงกับงาน AI/ML อย่างไร? | • การออกแบบระบบ AI ภายใต้ข้อจำกัดจริง (Hardware, Cost, Latency, License)<br>• ความเข้าใจลึกซึ้งเรื่อง Data Pipeline, Distribution Drift และ Metric Honesty<br>• ทักษะการ Deployment จริง (ONNX Runtime, C# Interop, FastAPI Microservice) |

---

## 🌟 4. STAR Interview Behavioral Guide (บทตอบสัมภาษณ์แบบ STAR)

### 1) Situation (บริบท)
> *"โจทย์ของโปรเจกต์นี้เริ่มต้นจากความต้องการระบบอ่านป้ายทะเบียนรถยนต์สำหรับกล้องวงจรปิดบริเวณด่านตรวจและประตูอัจฉริยะ ซึ่งต้องอ่านได้ทั้งป้ายทะเบียนของไทยและลาวที่มีโครงสร้างต่างกันอย่างสิ้นเชิง"*
>
> *"ความท้าทายสำคัญคือ ข้อจำกัดด้านระบบปฏิบัติการ: ฮาร์ดแวร์ปลายทางเป็นเครื่อง Server CPU ทั่วไป ไม่มี GPU ราคาแพง, ระบบหลังบ้านเดิมของลูกค้าพัฒนาด้วย C# (.NET), และต้องประมวลผลให้ได้ผลลัพธ์แบบ Real-time ต่ำกว่า 500 ms"*

### 2) Task (เป้าหมายและความรับผิดชอบ)
> *"หน้าที่หลักของผมคือ ออกแบบและพัฒนา AI Pipeline ตั้งแต่ต้นน้ำถึงปลายน้ำ: ตรวจจับตำแหน่งป้าย, ดัดระนาบ 4 มุม (Perspective Rectification), คัดแยกประเทศ, และอ่านตัวอักษรพร้อมชื่อจังหวัด โดยตั้งเป้าหมายว่าโมเดลต้องมีขนาดกะทัดรัด, ค่าความแม่นยำสูงในภาพเบลอ, ไม่มีข้อจำกัดด้านลิขสิทธิ์เชิงพาณิชย์ (หลีกเลี่ยง AGPL), และแปลงเป็น ONNX ได้อย่างสมบูรณ์"*

### 3) Action (การลงมือทำและเทคนิคทางวิศวกรรม)
> *"ผมแบ่งการแก้ปัญหาออกเป็น 4 ส่วนหลัก:*
> 1. **Decoupled Multi-Stage Architecture:** แทนที่จะใช้โมเดลใหญ่ตัวเดียว ผมแยกหน้าที่ให้โมเดลขนาดเล็กที่มีประสิทธิภาพสูง เช่น ใช้ PicoDet-S / D-FINE สำหรับหาตำแหน่งป้าย, ใช้ MobileNetV3 Regressor หาพิกัด 4 มุมเพื่อ Warp ระนาบป้ายให้แบนตรงระดับ Sub-pixel, และใช้ MobileNetV3 แยกสัญชาติไทย-ลาว
> 2. **Handling Lao Inverted Layout & Domain Logic:** ป้ายลาวมีเลย์เอาต์กลับด้านกับไทย (จังหวัดอยู่บน ตัวอักษรอยู่ล่าง) ผมสร้างกระบวนการ Flip-and-Detect และตั้งกฎ Spatial Zone Masking บังคับพยัญชนะอยู่ซ้ายและตัวเลขอยู่ขวา พร้อมคัดพยัญชนะลาวที่ไม่ได้ใช้งานออก 4 ตัวเพื่อตัดปัญหาภาพหลอน (Hallucination)
> 3. **Train-Serve Preprocessing Contract:** พบปัญหาโมเดลทายจังหวัดผิดเมื่อใส่ขอบดำ (Black Padding) ผมจึงสร้างฟังก์ชัน `GrayscaleSmartResize` ที่ดึงค่าสีเฉลี่ยจาก 4 มุมของภาพมาเติมขอบแทน ทำให้ Distribution ของภาพตอน Train และ Serve ตรงกันอย่างสมบูรณ์
> 4. **Dual-Engine Fusion & Real-Time Voting:** รวมผลลัพธ์ระหว่างโมเดลตัดกล่องตัวอักษร (Char Box Detector) กับโมเดล ResNet-CRNN CTC OCR โดยมีกฎไวยากรณ์กรมการขนส่งฯ (DLT Rules) ควบคุม และสร้างระบบ Majority Voting รวมผลลัพธ์ 3 เฟรมของรถคันเดียวกันเพื่อขจัด Noise จากการเคลื่อนที่"*

### 4) Result (ผลลัพธ์เชิงประจักษ์)
> *"ผลลัพธ์ที่ได้คือ ระบบสามารถรัน End-to-End บน CPU ด้วยเวลาเฉลี่ยเพียง **75–95 ms ต่อภาพ** (เร็วกว่าเป้าหมาย 500 ms ถึง 5 เท่า):*
> * คัดแยกป้ายไทยและลาวถูกต้อง **99.58%**
> * อ่านชื่อจังหวัดไทยได้แม่นยำ **92.47%** บนภาพจริง
> * ความแม่นยำตัวอักษรและหมวดป้ายไทย **81.80%**
> * อ่านป้ายและแขวงของลาวได้ความแม่นยำมากกว่า **80%** (Lao OCR CER ต่ำเพียง 1.02%)
> * ตัวโมเดล PicoDet-S ถูกแปลงเป็น ONNX Opset 18 และส่งมอบให้วิศวกรซอฟต์แวร์นำไปเชื่อมต่อทดสอบกับระบบ C# (.NET) ได้อย่างราบรื่นโดยไม่ต้องคอมไพล์ C++ Plugin เพิ่มเติม"*

---

## 🛡️ 5. Interview Defense (คำตอบสำหรับคำถามยากๆ ที่อาจเจอ)

### Q1: "ทำเล่นๆ หรือเปล่า? ทำไมถึงทำโปรเจกต์นี้?"
* **แนวทางตอบ:**
  > *"ไม่ใช่การทำเล่นๆ ครับ แต่เป็นการทำ **Engineering Feasibility PoC (Proof-of-Concept)** เพื่อศึกษาการแก้ปัญหา Computer Vision ในสภาวะแวดล้อมจริงที่มีข้อจำกัด (Resource-Constrained Edge Environment)*  
  > *ถ้าเป็นการทำเล่นๆ ผมคงดาวน์โหลดโมเดลสำเร็จรูปตัวใหญ่ๆ หรือเรียก Cloud API มาต่อก็จบ แต่ที่ผมออกแบบเป็น Multi-Stage เพราะต้องตอบโจทย์เรื่อง CPU Latency ต่ำกว่า 100 ms, ต้องแปลงเป็น ONNX รันบน C# ได้, และต้องไม่มีปัญหาลิขสิทธิ์ AGPL ซึ่งเป็นข้อจำกัดจริงในเชิงพาณิชย์ครับ"*

### Q2: "ทำไมไม่ใช้ YOLOv8 / YOLOv11 แบบ End-to-End ตัวเดียวไปเลย?"
* **แนวทางตอบ:**
  > *"มี 2 เหตุผลหลักครับ:*  
  > 1. **Licensing Constraint:** Ultralytics YOLO ใช้สัญญาอนุญาต AGPL-3.0 ซึ่งมีข้อผูกมัดทางกฎหมายสูงสำหรับงานเชิงพาณิชย์ ผมจึงเลือกใช้ PicoDet (Apache-2.0) และ D-FINE (MIT) แทน  
  > 2. **Perspective Distortion & Small Font Detail:** การใช้โมเดลกล่องหมุน (OBB) 1-Stage ตัวป้ายจะกินพื้นที่เพียง 3-5% ของภาพ 1080p ทำให้ตัวอักษรภาษาไทยและสระเล็กๆ เสียรายละเอียด แต่การตัดออกมาทำ Homography Warp สู่ภาพขนาด 320x160 ก่อน ทำให้โมเดลขั้นถัดไปมองเห็นตัวอักษรชัดเจนและแม่นยำกว่ามากครับ"*

### Q3: "หา Dataset มาจากไหน และจัดการกับความไม่สมดุลของข้อมูลอย่างไร?"
* **แนวทางตอบ:**
  > *"ผมรวบรวมข้อมูลภาพจากแหล่งเปิดและกล้องจราจรสาธารณะ ตัดเฉพาะส่วนป้ายทะเบียนเพื่อไม่ให้มีข้อมูลส่วนบุคคล (PII) ของเจ้าของรถ*  
  > *ปัญหาใหญ่คือ Class Imbalance เช่น กรุงเทพมหานครมีภาพมากกว่าจังหวัดเล็กๆ นับร้อยเท่า ผมแก้ปัญหาโดยการใช้ `WeightedRandomSampler` ในการสุ่มตอนเทรน, เสริมข้อมูลสังเคราะห์ (Font Synthesis) สำหรับหมวดอักษรที่พบน้อย, และใช้ Augmentation จำลองแสงสะท้อน การเบลอ และการบิดมุมมองครับ"*
