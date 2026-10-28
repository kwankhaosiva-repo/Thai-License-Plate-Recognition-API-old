using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model: Lao Province Detector 1-Stage (PicoDet-S Lao v6 ONNX, 320x320, 19 Classes)
    /// Direct Plate-Crop Object Detector: predicts bounding box and 19 Lao province classes
    /// directly from a 320x160 rectified plate image without external crop dependency.
    ///
    /// Preprocessing Contract (Serve == Train):
    /// 1. Direct Resize to 320x320 RGB (Linear Interpolation).
    /// 2. ImageNet Normalization (0-255 scale):
    ///    Mean: [123.675, 116.28, 103.53], Std: [58.395, 57.12, 57.375].
    /// 3. Output shape: [1, 2125, 23] -> 4 bbox coords [x1, y1, x2, y2] + 19 class scores (c0..c18).
    /// </summary>
    public class LaoProvinceDetector1Stage : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        // ImageNet normalization constants in 0-255 scale (RGB order)
        private static readonly float[] Mean = { 123.675f, 116.28f, 103.53f };
        private static readonly float[] Std  = { 58.395f, 57.12f, 57.375f };

        /// <summary>
        /// 19 Lao Provinces Mapping (Matches weights/province_map_lao.json exactly)
        /// </summary>
        public static readonly Dictionary<int, string> ProvinceMap = new()
        {
            { 0, "ນະຄອນຫຼວງວຽງຈັນ" }, // Vientiane Capital (Modern full name)
            { 1, "ຜົ້ງສາລີ" },
            { 2, "ຫຼວງນ້ຳທາ" },
            { 3, "ອຸດົມໄຊ" },
            { 4, "ບໍ່ແກ້ວ" },
            { 5, "ຫຼວງພະບາງ" },
            { 6, "ຫົວພັນ" },
            { 7, "ໄຊຍະບູລີ" },
            { 8, "ຊຽງຂວາງ" },
            { 9, "ວຽງຈັນ" },           // Vientiane Province
            { 10, "ບໍລິຄຳໄຊ" },
            { 11, "ຄຳມ່ວນ" },
            { 12, "ສະຫວັນນະເຂດ" },
            { 13, "ສາລະວັນ" },
            { 14, "ເຊກອງ" },
            { 15, "ຈຳປາສັກ" },
            { 16, "ອັດຕະປື" },
            { 17, "ໄຊສົມບູນ" },
            { 18, "ກຳແພງນະຄອນ" }      // Kamphaeng Nakhon (Historic/Short variant)
        };

        public class DetectionResult
        {
            public string Province { get; set; } = string.Empty;
            public int ClassId { get; set; }
            public float Confidence { get; set; }
            public Rect BannerBox { get; set; }
        }

        public LaoProvinceDetector1Stage(string onnxPath)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Detects and classifies the Lao province directly from a rectified plate (e.g. 320x160).
        /// Returns the single best detection exceeding the confidence threshold.
        /// </summary>
        /// <param name="rectifiedPlate">BGR plate image (typically 320x160)</param>
        /// <param name="confThreshold">Confidence threshold (default: 0.20)</param>
        /// <returns>Tuple of (Province Name, Confidence Percentage, Banner Box) or null if not detected</returns>
        public (string Province, float Conf, Rect BannerBox)? DetectProvince(Mat rectifiedPlate, float confThreshold = 0.20f)
        {
            var results = DetectAllProvinces(rectifiedPlate, confThreshold, iouThreshold: 0.40f);
            if (results.Count == 0)
                return null;

            var best = results[0];
            return (best.Province, best.Confidence, best.BannerBox);
        }

        /// <summary>
        /// Detects all province banner candidates on the plate with Non-Maximum Suppression (NMS).
        /// </summary>
        /// <param name="rectifiedPlate">BGR plate image</param>
        /// <param name="confThreshold">Confidence threshold</param>
        /// <param name="iouThreshold">NMS IoU threshold (default: 0.40)</param>
        /// <returns>List of candidate detections sorted by confidence descending</returns>
        public List<DetectionResult> DetectAllProvinces(Mat rectifiedPlate, float confThreshold = 0.20f, float iouThreshold = 0.40f)
        {
            if (rectifiedPlate == null || rectifiedPlate.Empty() || rectifiedPlate.Width < 10 || rectifiedPlate.Height < 10)
                return new List<DetectionResult>();

            int origW = rectifiedPlate.Width;
            int origH = rectifiedPlate.Height;

            // 1. Resize to 320x320 RGB (Exact match with training imgsz=320)
            using var rgb320 = new Mat();
            Cv2.Resize(rectifiedPlate, rgb320, new Size(320, 320), 0, 0, InterpolationFlags.Linear);
            Cv2.CvtColor(rgb320, rgb320, ColorConversionCodes.BGR2RGB);

            // 2. ImageNet Normalization: (RGB - Mean) / Std
            var tensor = new DenseTensor<float>(new[] { 1, 3, 320, 320 });
            for (int y = 0; y < 320; y++)
            {
                for (int x = 0; x < 320; x++)
                {
                    Vec3b p = rgb320.At<Vec3b>(y, x);
                    tensor[0, 0, y, x] = (p.Item0 - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = (p.Item1 - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = (p.Item2 - Mean[2]) / Std[2];
                }
            }

            // 3. ONNX Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>(); // [1, 2125, 23]

            int numCandidates = output.Dimensions[1];
            int numClasses = ProvinceMap.Count; // 19 classes (0..18)
            float scaleX = origW / 320.0f;
            float scaleY = origH / 320.0f;

            var candidates = new List<DetectionResult>();

            for (int i = 0; i < numCandidates; i++)
            {
                // Find class with highest probability for this anchor candidate
                float maxClsScore = -1f;
                int currentCls = -1;
                for (int c = 0; c < numClasses; c++)
                {
                    float clsProb = output[0, i, 4 + c];
                    if (clsProb > maxClsScore)
                    {
                        maxClsScore = clsProb;
                        currentCls = c;
                    }
                }

                if (maxClsScore >= confThreshold && currentCls >= 0)
                {
                    int x1 = Math.Max(0, (int)Math.Round(output[0, i, 0] * scaleX));
                    int y1 = Math.Max(0, (int)Math.Round(output[0, i, 1] * scaleY));
                    int x2 = Math.Min(origW, (int)Math.Round(output[0, i, 2] * scaleX));
                    int y2 = Math.Min(origH, (int)Math.Round(output[0, i, 3] * scaleY));

                    int bw = Math.Max(1, x2 - x1);
                    int bh = Math.Max(1, y2 - y1);

                    candidates.Add(new DetectionResult
                    {
                        Province = ProvinceMap.TryGetValue(currentCls, out var name) ? name : $"Class_{currentCls}",
                        ClassId = currentCls,
                        Confidence = maxClsScore * 100.0f,
                        BannerBox = new Rect(x1, y1, bw, bh)
                    });
                }
            }

            // 4. Apply Non-Maximum Suppression (NMS)
            return ApplyNms(candidates, iouThreshold);
        }

        private static List<DetectionResult> ApplyNms(List<DetectionResult> candidates, float iouThreshold)
        {
            if (candidates.Count == 0) return candidates;

            var sorted = candidates.OrderByDescending(c => c.Confidence).ToList();
            var selected = new List<DetectionResult>();

            foreach (var current in sorted)
            {
                bool shouldKeep = true;
                foreach (var kept in selected)
                {
                    float iou = ComputeIoU(current.BannerBox, kept.BannerBox);
                    if (iou > iouThreshold)
                    {
                        shouldKeep = false;
                        break;
                    }
                }

                if (shouldKeep)
                {
                    selected.Add(current);
                }
            }

            return selected;
        }

        private static float ComputeIoU(Rect a, Rect b)
        {
            int ix1 = Math.Max(a.Left, b.Left);
            int iy1 = Math.Max(a.Top, b.Top);
            int ix2 = Math.Min(a.Right, b.Right);
            int iy2 = Math.Min(a.Bottom, b.Bottom);

            int interW = Math.Max(0, ix2 - ix1);
            int interH = Math.Max(0, iy2 - iy1);
            float interArea = interW * interH;

            float unionArea = (a.Width * a.Height) + (b.Width * b.Height) - interArea;
            return unionArea > 0 ? interArea / unionArea : 0f;
        }

        public void Dispose() => _session?.Dispose();
    }
}
