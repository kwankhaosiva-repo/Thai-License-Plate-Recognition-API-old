# 🛣️ คู่มืออธิบายหลักการและสูตรคำนวณระบบตรวจจับการวิ่งคร่อมเลน
## (Highway Lane Straddling & Departure Detection Algorithm Guide)

เอกสารฉบับนี้จัดทำขึ้นเพื่ออธิบายหลักการทางคณิตศาสตร์, ฟิสิกส์ยานยนต์ (Vehicle Kinematics), เรขาคณิตภาพฉาย (Projective Geometry), วัตถุประสงค์ และสูตรคำนวณของแต่ละฟังก์ชันหลักในระบบตรวจจับการวิ่งคร่อมเลน (`src/lane_engine.py` และ `src/LaneStraddlingEngine.cs`) อย่างละเอียด

---

## 📌 สรุปภาพรวมกระบวนการประมวลผล (Pipeline Flowchart)

```
[ภาพจากกล้อง CCTV 2D] 
        │
        ▼
[Function 1: compute_homography] ──► คำนวณ Matrix H แปลงสู่ระนาบถนน 3.5m x 35.0m
        │
[Function 2: auto_square_calibration] ──► ขจัด Perspective Shear จัดระนาบลึก 90 องศา
        │
        ▼
[Stage 1: Vehicle Detector] ──► ค้นหา Bounding Box ของตัวรถ (LibreDFINE-Nano)
        │
        ▼
[Function 3: detect_tires_on_vehicle] ──► ตรวจจับหัวเก๋ง (cab_face) และล้อหน้า (front_wheel)
        │
        ▼
[Function 4: estimate_vehicle_ground_contact] ──► คำนวณจุดล้อคู่หน้า (Dynamic Track Width + Ground Y-Anchor)
        │
        ▼
[Function 5: assess_lane_straddling] ──► ประเมินการเหยียบทับเส้นด้วย Finite Tire Contact Patch (14 cm)
        │
        ▼
[ผลลัพธ์: IN_LANE / STRADDLE_LEFT / STRADDLE_RIGHT / OUT_OF_LANE พร้อมระยะเซนติเมตร]
```

---

## 1. ฟังก์ชัน `compute_homography()`

### 1.1 หลักการที่ใช้ (Principle)
อาศัยหลักการทางเรขาคณิตภาพฉาย **Planar Homography Transformation ($\mathbb{P}^2 \to \mathbb{P}^2$)** ซึ่งระบุว่า สำหรับระนาบ 2 ระนาบในโลก 3 มิติ (ระนาบพื้นผิวถนน และระนาบเซนเซอร์ของกล้อง) จะมีความสัมพันธ์เชิงเส้นแบบโปรเจกทีฟที่สามารถแปลงจุดคู่ใดๆ ไป-กลับได้ผ่านเมทริกซ์ขนาด $3 \times 3$ ที่เรียกว่า **Homography Matrix ($\mathbf{H}$)**

### 1.2 ฟังก์ชันนี้ทำไปเพื่ออะไร (Objective)
- แปลงพิกัดภาพเฉียง 2D จากกล้องวงจรปิด (ซึ่งมีอัตราส่วนบิดเบี้ยวตามความลึก) ให้กลายเป็นภาพมุมมองบนระนาบถนนจริง (Bird's Eye View - BEV) ในหน่วยเมตริกจริง ($3.5\text{ m} \times 35.0\text{ m}$)
- ทำให้เราสามารถวัดระยะห่างระหว่างล้อรถกับเส้นเลนได้เป็นหน่วย **"เซนติเมตรจริง"** แทนที่จะเป็นพิกเซลหลอก

### 1.3 สูตรคำนวณ (Mathematical Formulation)
พิกัดจุดบนภาพกล้อง $(x, y)$ แปลงไปยังพิกัดบนระนาบพื้นถนน BEV $(x', y')$:

$$s \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

เมื่อทำการกระจายสมการเพื่อหาพิกัด Cartesian $x'$ และ $y'$:

$$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}$$

$$y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}$$

เมทริกซ์ $\mathbf{H}$ คำนวณโดยใช้วิธี **Direct Linear Transformation (DLT)** จากจุดอ้างอิงบนพื้นถนน 4 จุดคู่สมนัย:
- **จุดต้นทาง (Source 2D Camera)**: สี่เหลี่ยมคอร์ริดอร์บนถนน $G_1, G_2, G_3, G_4$
- **จุดปลายทาง (Destination Metric BEV)**: $[(0, 0), (W_{\text{bev}}-1, 0), (W_{\text{bev}}-1, H_{\text{bev}}-1), (0, H_{\text{bev}}-1)]$

สำหรับการแปลงกลับจาก BEV มายังระนาบภาพกล้อง (Inverse Homography):

$$\mathbf{p}_{\text{cam}} = \mathbf{H}^{-1} \mathbf{p}_{\text{bev}}$$

---

## 2. ฟังก์ชัน `auto_square_calibration()`

### 2.1 หลักการที่ใช้ (Principle)
อาศัยหลักการ **Transverse Orthogonality ($90^\circ$ Perpendicular Slice)** และการกำจัด **Depth Disparity (Longitudinal Shear)**:
บนถนนจริง แนวเส้นตัดขวางถนน (Transverse Line) เชื่อมระหว่างขอบทางซ้ายและขวา จะต้องทำมุมตั้งฉาก $90^\circ$ กับทิศทางการเดินรถเสมอ

### 2.2 ฟังก์ชันนี้ทำไปเพื่ออะไร (Objective)
- เมื่อผู้ใช้ลากจุด 4 จุดด้วยมือเปล่า มักเกิดความคลาดเคลื่อน โดยคลิกจุดฝั่งขวา ($G_3$) ลึกเข้าไปในถนนมากกว่าฝั่งซ้าย ($G_4$) ถึง 50–60 พิกเซล
- ความคลาดเคลื่อนในแกนลึกนี้ จะบังคับให้ Homography Matrix เกิดแรงบิดระนาบ (Shear Angle $15^\circ - 18^\circ$) ส่งผลให้เมื่อโปรเจกต์ล้ออีกข้างกลับมา เพลาล้อจะเฉียงลอยขึ้นไปบนกระจังหน้ารถ
- ฟังก์ชันนี้ช่วยเกลี่ยระดับแกน $Y$ ของจุดคู่หน้า ($G_1 \leftrightarrow G_2$) และจุดคู่หลัง ($G_4 \leftrightarrow G_3$) ให้ขนานกับแนวตัดขวางถนนจริง ขจัดปัญหาเพลาล้อเอียง 100% โดยผู้ใช้ไม่ต้องกะด้วยสายตา

### 2.3 สูตรคำนวณ (Mathematical Formulation)
สำหรับจุดพิกัด 4 จุด $G_1(x_1, y_1), G_2(x_2, y_2), G_3(x_3, y_3), G_4(x_4, y_4)$:

1. **ระดับความลึกฝั่งไกล (Far Depth Baseline):**
   $$y_{\text{far}} = \frac{y_{G_1} + y_{G_2}}{2}$$

2. **ระดับความลึกฝั่งใกล้ (Near Depth Baseline):**
   $$y_{\text{near}} = \frac{y_{G_4} + y_{G_3}}{2}$$

3. **พิกัดสี่เหลี่ยมหลังปรับจูน (Auto-Squared Quad):**
   $$G'_1 = (x_{G_1}, \; y_{\text{far}})$$
   $$G'_2 = (x_{G_2}, \; y_{\text{far}})$$
   $$G'_3 = (x_{G_3}, \; y_{\text{near}})$$
   $$G'_4 = (x_{G_4}, \; y_{\text{near}})$$

ผลลัพธ์คือความชันของเส้นตัดขวางถนน $\Delta y / \Delta x = 0$ ทำให้มุมบิด $\theta_{\text{shear}} = 0^\circ$

---

## 3. ฟังก์ชัน `detect_tires_on_vehicle()`

### 3.1 หลักการที่ใช้ (Principle)
อาศัยหลักการ **Two-Stage Spatial Hierarchy** และ **Scale-Aware Dynamic Margins**:
- ขั้นที่ 1 ตรวจจับกรอบหน้ารถ (`cab_face`)
- ขั้นที่ 2 จำกัดขอบเขตการค้นหาล้อหน้าเฉพาะบริเวณรอยสัมผัสถนนใต้ซุ้มล้อหน้าเท่านั้น

### 3.2 ฟังก์ชันนี้ทำไปเพื่ออะไร (Objective)
- ป้องกันไม่ให้ AI สับสนระหว่าง "ล้อรถจริง" กับ "เงาดำบนถนน", "ไฟตัดหมอก", "ล้อรถพ่วงด้านข้าง" หรือ "กระจกมองข้างที่ลอยอยู่กลางอากาศ"
- หากมองเห็นล้อชัดเจนเพียงข้างเดียว (อีกฝั่งถูกมุมมองบดบัง) ระบบจะเลือกเฉพาะล้อที่มี Confidence สูงสุดฝั่งที่เห็นชัด เพื่อใช้เป็น Anchor ที่แม่นยำ

### 3.3 สูตรคำนวณ (Mathematical Formulation)
1. **ขอบเขตระยะเผื่อตามสเกลหน้ารถจริง (Scale-Aware Dynamic Margins):**
   $$\Delta x_{\text{margin}} = \max\left(35.0, \; 0.30 \times W_{\text{cab}}\right)$$
   $$\Delta y_{\text{margin}} = \max\left(45.0, \; 0.35 \times H_{\text{cab}}\right)$$

2. **เงื่อนไขการยอมรับกรอบล้อหน้า (Containment Spatial Filter):**
   $$x_{\text{cab\_min}} - \Delta x_{\text{margin}} \le x_{\text{wheel\_center}} \le x_{\text{cab\_max}} + \Delta x_{\text{margin}}$$
   $$y_{\text{cab\_bottom}} - \Delta y_{\text{margin}} \le y_{\text{wheel\_bottom}} \le y_{\text{cab\_bottom}} + 25.0$$

3. **เงื่อนไขคัดกรองล้อเด่น (Dominant Wheel Gating):**
   $$\text{If } \text{Conf}_{\text{dominant}} \ge 0.40 \quad \text{and} \quad \text{Conf}_{\text{opposite}} < 0.35 \implies \text{ใช้เฉพาะล้อเด่นเป็น Anchor}$$

---

## 4. ฟังก์ชัน `estimate_vehicle_ground_contact()`

### 4.1 หลักการที่ใช้ (Principle)
อาศัยหลักการ **Automotive Kinematic Invariants (สัจธรรมโครงสร้างยานยนต์)**:
1. ระยะห่างระหว่างกึ่งกลางล้อคู่หน้า (Front Wheel Track Width: $W_{\text{track}}$) จะต้องอยู่ภายในซุ้มล้อเสมอ ($W_{\text{track}} \approx 0.82\text{--}0.88 \times W_{\text{cab}}$)
2. จุดสัมผัสพื้นของล้อหน้าทั้ง 2 ข้าง ต้องวางตัวอยู่บนระนาบพื้นดินระดับเดียวกับกันชนหน้ารถ

### 4.2 ฟังก์ชันนี้ทำไปเพื่ออะไร (Objective)
- **แก้ปัญหาภาพที่ 17 (ล้อทะลุตัวถัง):** รถกระบะตู้ทึบหรือคอกสูงที่โมเดลเข้าใจว่าเป็น `truck` จะไม่ถูกยัดเยียดฐานล้อ 2.05 ม. ของรถ 10 ล้ออีกต่อไป แต่จะปรับฐานล้อตามความกว้างหน้ารถจริงใน BEV ($\approx 1.49\text{ m}$)
- **แก้ปัญหาภาพที่ 3 (เพลาล้อชี้ขึ้นกระจังหน้า):** ตรึงระดับแกน $Y$ ของล้อที่โปรเจกต์กลับมาให้อยู่ในระนาบพื้นของกันชนหน้ารถเสมอ

### 4.3 สูตรคำนวณ (Mathematical Formulation)

1. **คำนวณความกว้างหน้ารถในระนาบ BEV:**
   $$\mathbf{p}_{\text{bev\_l}} = \mathbf{H} \begin{bmatrix} x_{\text{cab\_min}} \\ y_{\text{cab\_bottom}} \\ 1 \end{bmatrix}, \quad \mathbf{p}_{\text{bev\_r}} = \mathbf{H} \begin{bmatrix} x_{\text{cab\_max}} \\ y_{\text{cab\_bottom}} \\ 1 \end{bmatrix}$$
   $$W_{\text{cab\_m}} = \frac{|\mathbf{p}_{\text{bev\_r}}.x - \mathbf{p}_{\text{bev\_l}}.x|}{\text{PPM}_x}$$

2. **การปรับฐานล้อพลวัต (Dynamic Track Width Scaling):**
   $$W_{\text{track\_m}} = \begin{cases} \min\left(2.05, \; \max(1.40, \; W_{\text{cab\_m}} \times 0.88)\right) & \text{ถ้า } W_{\text{cab\_m}} < 1.95\text{ m (กระบะ/ตู้/เล็ก)} \\ \min\left(2.05, \; W_{\text{cab\_m}} \times 0.90\right) & \text{ถ้า } W_{\text{cab\_m}} \ge 1.95\text{ m (รถบรรทุกใหญ่/บัส)} \end{cases}$$
   $$W_{\text{track\_bev}} = W_{\text{track\_m}} \times \text{PPM}_x$$

3. **การคำนวณตำแหน่งล้ออีกฝั่ง (Opposite Wheel Projection):**
   สมมุติพบล้อขวาเป็น Anchor ที่พิกัดกล้อง $(t_x, t_y)$:
   $$\mathbf{p}_{\text{bev\_r}} = \mathbf{H} \begin{bmatrix} t_x \\ t_y \\ 1 \end{bmatrix}$$
   $$\mathbf{p}_{\text{bev\_l}} = \begin{bmatrix} \mathbf{p}_{\text{bev\_r}}.x - W_{\text{track\_bev}} \\ \mathbf{p}_{\text{bev\_r}}.y \end{bmatrix}$$
   $$\mathbf{p}_{\text{cam\_proj}} = \mathbf{H}^{-1} \mathbf{p}_{\text{bev\_l}}$$

4. **การล็อกระดับพื้นกันชนและขอบเขตตัวถัง (Bumper Y-Anchor & Lateral Clamp):**
   $$y_{\text{left\_final}} = \text{clamp}\left(\mathbf{p}_{\text{cam\_proj}}.y, \; y_{\text{cab\_bottom}} - 8.0, \; y_{\text{cab\_bottom}} + 8.0\right)$$
   $$x_{\text{left\_final}} = \text{clamp}\left(\mathbf{p}_{\text{cam\_proj}}.x, \; x_{\text{cab\_min}} - 10.0, \; x_{\text{cab\_max}} + 10.0\right)$$

---

## 5. ฟังก์ชัน `assess_lane_straddling()`

### 5.1 หลักการที่ใช้ (Principle)
อาศัยหลักการ **Finite Tire Contact Patch Dynamics (ฟิสิกส์ขนาดหน้ายางจริง)**:
- ในทางวิศวกรรมจราจร ยางรถยนต์ไม่ใช่จุด 0 cm
- ยางรถบรรทุกขนาด 295/80R22.5 หรือ 11R22.5 มีหน้ากว้างสัมผัสพื้น $28 - 30\text{ cm}$
- จุดกึ่งกลางล้อถึงขอบผิวยางด้านนอกมีระยะครึ่งหน้ายาง $W_{\text{tire\_half}} = 14.0\text{ cm}$

### 5.2 ฟังก์ชันนี้ทำไปเพื่ออะไร (Objective)
- **แก้ปัญหาภาพที่ 33 (รถอยู่ชิดเส้นแต่แจ้งว่าไม่ทับ):** เมื่อจุดกึ่งกลางล้ออยู่ห่างจากเส้นเลน $1.6\text{ cm}$ แต่หน้ายางกว้าง $14\text{ cm}$ แสดงว่าขอบผิวดอกยางได้เหยียบข้ามเส้นไปแล้ว $12.4\text{ cm}$
- ฟังก์ชันนี้เปลี่ยนจากการตรวจจุด 0 cm มาเป็นการตรวจรอยสัมผัสของหน้ายางจริง เพื่อให้สอดคล้องกับกฎหมายจราจรที่นับว่า "การเหยียบทับเส้น = การวิ่งคร่อมเลน"

### 5.3 สูตรคำนวณ (Mathematical Formulation)

1. **การ Interpolate ตำแหน่งพิกัด X ของเส้นเลน ณ ความลึกของล้อ (BEV Space):**
   $$x_{\text{line\_left}}(y_{\text{wheel}}) = x_{L1} + \frac{y_{\text{wheel}} - y_{L1}}{y_{L2} - y_{L1}} \left(x_{L2} - x_{L1}\right)$$
   $$x_{\text{line\_right}}(y_{\text{wheel}}) = x_{R1} + \frac{y_{\text{wheel}} - y_{R1}}{y_{R2} - y_{R1}} \left(x_{R2} - x_{R1}\right)$$

2. **ระยะห่างจากจุดกึ่งกลางล้อถึงเส้นเลน (หน่วยเซนติเมตร):**
   $$d_{\text{left\_center\_cm}} = \frac{\mathbf{p}_{\text{bev\_left}}.x - x_{\text{line\_left}}(y_{\text{bev\_left}})}{\text{PPM}_x / 100}$$
   $$d_{\text{right\_center\_cm}} = \frac{x_{\text{line\_right}}(y_{\text{bev\_right}}) - \mathbf{p}_{\text{bev\_right}}.x}{\text{PPM}_x / 100}$$

3. **ระยะเคลียร์ของขอบผิวยางด้านนอก (Tire Edge Clearance):**
   $$W_{\text{tire\_half}} = \begin{cases} 14.0\text{ cm} & \text{สำหรับรถบรรทุก/รถบัส (Truck/Bus)} \\ 10.0\text{ cm} & \text{สำหรับรถยนต์/กระบะ (Car/Van)} \end{cases}$$

   $$\text{Clearance}_{\text{left}} = d_{\text{left\_center\_cm}} - W_{\text{tire\_half}}$$
   $$\text{Clearance}_{\text{right}} = d_{\text{right\_center\_cm}} - W_{\text{tire\_half}}$$

4. **การตัดสินสถานะการคร่อมเลน (Straddle Decision Matrix):**
   $$\text{Straddle Left} \iff \text{Clearance}_{\text{left}} < -\text{Tolerance}_{\text{cm}}$$
   $$\text{Straddle Right} \iff \text{Clearance}_{\text{right}} < -\text{Tolerance}_{\text{cm}}$$

   - หาก $\text{Straddle Left} = \text{True}$ และ $\text{Straddle Right} = \text{True} \implies \mathbf{OUT\_OF\_LANE}$
   - หาก $\text{Straddle Left} = \text{True} \implies \mathbf{STRADDLE\_LEFT}$ พร้อมระยะล้ำเส้น $D = |\text{Clearance}_{\text{left}}|\text{ cm}$
   - หาก $\text{Straddle Right} = \text{True} \implies \mathbf{STRADDLE\_RIGHT}$ พร้อมระยะล้ำเส้น $D = |\text{Clearance}_{\text{right}}|\text{ cm}$
   - นอกนั้น $\implies \mathbf{IN\_LANE}$ พร้อมรายงานระยะห่างขอบยางถึงเส้นอย่างปลอดภัย

---

## 6. สรุปพารามิเตอร์คงที่เชิงกายภาพ (Physical Constants Summary Table)

| ตัวแปร | ค่าตัวเลข | หน่วย | เหตุผลทางกายภาพและวิศวกรรม |
| :--- | :---: | :---: | :--- |
| `WorldWidthM` | $3.50$ | เมตร | ความกว้างมาตรฐานของช่องจราจรทางหลวงไทย (DOH Standard Lane) |
| `WorldLengthM` | $35.0$ | เมตร | ระยะคอร์ริดอร์ตรวจจับความปลอดภัยของด่านชั่งและทางหลวง |
| `TruckBaseTrackWidth` | $2.05$ | เมตร | ระยะกึ่งกลางล้อหน้ามาตรฐานของรถบรรทุก 10 ล้อ (Hino 500 / Isuzu Giga) |
| `PickupTrackWidthRatio`| $0.88$ | เท่า | อัตราส่วนฐานล้อต่อความกว้างตัวถังรถกระบะ ($1.70\text{m} \times 0.88 \approx 1.50\text{m}$) |
| `TruckTireHalfWidth` | $14.0$ | เซนติเมตร | ครึ่งหนึ่งของขนาดยางรถบรรทุกมาตรฐานเรเดียล 295/80R22.5 ($28\text{ cm}$) |
| `CarTireHalfWidth` | $10.0$ | เซนติเมตร | ครึ่งหนึ่งของขนาดยางรถยนต์ทั่วไป ($20\text{ cm}$) |
| `BumperGroundMargin` | $\pm 8.0$ | พิกเซล | ระยะล็อกแกน Y ของล้อให้อยู่ในระนาบพื้นกันชนหน้ารถ ป้องกันเพลาลอย |
