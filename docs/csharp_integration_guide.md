# C# ONNX Integration Guide — Thai License Plate Detector

Run **D-FINE Nano / Small** (`plate_detector_dfine_nano.onnx`) directly from C# with
`Microsoft.ML.OnnxRuntime`. No Python, no LibreYOLO, no wrapper framework needed.

---

## 1. NuGet Packages

```bash
dotnet add package Microsoft.ML.OnnxRuntime   # CPU inference (all platforms)
dotnet add package OpenCvSharp4               # image I/O & preprocessing
dotnet add package OpenCvSharp4.runtime.osx   # change to .win or .ubuntu as needed
```

For GPU on Windows use `Microsoft.ML.OnnxRuntime.DirectML` instead.

---

## 2. ONNX Model Spec

| Item | Value |
|---|---|
| Input name | `images` |
| Input shape | `[1, 3, 640, 640]` float32 |
| Input range | `[0.0, 1.0]` (no ImageNet normalisation needed) |
| Output `pred_logits` | `[1, 300, 1]` — raw logit per candidate box |
| Output `pred_boxes` | `[1, 300, 4]` — `(cx, cy, w, h)` **normalised** (0–1) |

Post-processing steps:
1. `sigmoid(logit)` → confidence score
2. Filter by threshold (e.g. 0.35)
3. Convert `(cx, cy, w, h)` → `(x1, y1, x2, y2)` in pixel coordinates

---

## 3. Full C# Class

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

/// <summary>
/// Runs the exported D-FINE Nano/Small plate-detector ONNX model.
/// Thread-safe: one InferenceSession can be shared across threads.
/// </summary>
public sealed class DFinePlateDetector : IDisposable
{
    private const int ModelW = 640;
    private const int ModelH = 640;
    private const int MaxDets = 300;

    private readonly InferenceSession _session;

    public DFinePlateDetector(string onnxModelPath)
    {
        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        opts.AppendExecutionProvider_CPU();
        // Uncomment for GPU on Windows:
        // opts.AppendExecutionProvider_DML(0);
        _session = new InferenceSession(onnxModelPath, opts);
    }

    /// <summary>
    /// Detect license plates in the given BGR image (OpenCvSharp Mat).
    /// Returns bounding boxes in original image pixel coordinates.
    /// </summary>
    public List<(Rect Box, float Confidence)> Detect(Mat srcBgr, float confThreshold = 0.35f)
    {
        int origW = srcBgr.Width;
        int origH = srcBgr.Height;

        // 1. Preprocess: resize 640x640, BGR->RGB, normalise [0,1]
        using var resized = new Mat();
        Cv2.Resize(srcBgr, resized, new Size(ModelW, ModelH));
        Cv2.CvtColor(resized, resized, ColorConversionCodes.BGR2RGB);

        var tensor = new DenseTensor<float>(new[] { 1, 3, ModelH, ModelW });
        for (int y = 0; y < ModelH; y++)
        for (int x = 0; x < ModelW; x++)
        {
            Vec3b px = resized.At<Vec3b>(y, x);
            tensor[0, 0, y, x] = px.Item0 / 255f; // R
            tensor[0, 1, y, x] = px.Item1 / 255f; // G
            tensor[0, 2, y, x] = px.Item2 / 255f; // B
        }

        // 2. Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", tensor)
        };
        using var outputs = _session.Run(inputs);

        var logitsTensor = outputs.First(o => o.Name == "pred_logits").AsTensor<float>();
        var boxesTensor  = outputs.First(o => o.Name == "pred_boxes").AsTensor<float>();

        // 3. Post-process
        // pred_logits [1,300,1]: raw logit -> sigmoid -> confidence
        // pred_boxes  [1,300,4]: (cx, cy, w, h) normalised [0..1]
        var detections = new List<(Rect, float)>();
        for (int i = 0; i < MaxDets; i++)
        {
            float logit = logitsTensor[0, i, 0];
            float conf  = 1f / (1f + MathF.Exp(-logit)); // sigmoid

            if (conf < confThreshold) continue;

            // Decode normalised cx/cy/w/h -> pixel xyxy
            float cx = boxesTensor[0, i, 0] * origW;
            float cy = boxesTensor[0, i, 1] * origH;
            float bw = boxesTensor[0, i, 2] * origW;
            float bh = boxesTensor[0, i, 3] * origH;

            int x1 = Math.Clamp((int)(cx - bw / 2), 0, origW - 1);
            int y1 = Math.Clamp((int)(cy - bh / 2), 0, origH - 1);
            int x2 = Math.Clamp((int)(cx + bw / 2), 0, origW - 1);
            int y2 = Math.Clamp((int)(cy + bh / 2), 0, origH - 1);

            detections.Add((new Rect(x1, y1, x2 - x1, y2 - y1), conf));
        }

        detections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return detections;
    }

    public void Dispose() => _session?.Dispose();
}
```

---

## 4. Usage Example

```csharp
// Load once at startup
using var detector = new DFinePlateDetector("plate_detector_dfine_nano.onnx");

// Per-request (~25 ms on CPU)
using var image = Cv2.ImRead("car.jpg", ImreadModes.Color);
var plates = detector.Detect(image, confThreshold: 0.35f);

foreach (var (box, conf) in plates)
{
    Console.WriteLine($"Plate conf={conf:P1}  at ({box.X},{box.Y})  {box.Width}x{box.Height}px");

    // Crop for downstream OCR
    using var crop = new Mat(image, box);
    Cv2.ImWrite($"plate_{conf:F2}.jpg", crop);
}
```

---

## 5. FAQ

| Question | Answer |
|---|---|
| Need LibreYOLO in C#? | **No.** ONNX Runtime is self-contained. |
| Need Python at runtime? | **No.** Pure C#/.NET. |
| Need extra NMS step? | **No.** D-FINE applies NMS internally before ONNX export. |
| Use in ASP.NET Core? | **Yes.** Register `DFinePlateDetector` as a singleton in DI. |
| Windows / Linux / macOS? | **Yes.** `Microsoft.ML.OnnxRuntime` is fully cross-platform. |

---

## 6. Performance Tips

- Instantiate `DFinePlateDetector` **once** (startup ~500 ms, per-call ~25 ms CPU).
- `InferenceSession.Run()` is **thread-safe** — safe to call from multiple ASP.NET threads.
- Replace the pixel loop with `unsafe` pointer access for ~2× faster preprocessing.
- On Windows swap to `Microsoft.ML.OnnxRuntime.DirectML` for GPU with **zero code changes**.
