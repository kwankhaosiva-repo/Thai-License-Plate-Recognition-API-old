using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model: Lao Province Detector 1-Stage (PicoDet-S Lao v4 ONNX, 320x320)
    /// Direct Plate-Crop Object Detector: predicts bounding box and 18 Lao province classes
    /// directly from the flat 320x160 plate image without extra padding.
    /// </summary>
    public class LaoProvinceDetector1Stage : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        // ImageNet normalization constants in 0-255 scale
        private static readonly float[] Mean = { 123.675f, 116.28f, 103.53f };
        private static readonly float[] Std  = { 58.395f, 57.12f, 57.375f };

        // 18 Lao Provinces Mapping
        public static readonly Dictionary<int, string> ProvinceMap = new()
        {
            {0, "ນະຄອນຫຼວງວຽງຈັນ / ກຳແພງນະຄອນ"},
            {1, "ຜົ້ງສາລີ"},
            {2, "ຫຼວງນ້ຳທາ"},
            {3, "ອຸດົມໄຊ"},
            {4, "ບໍ່ແກ້ວ"},
            {5, "ຫຼວງພະບາງ"},
            {6, "ຫົວພັນ"},
            {7, "ໄຊຍະບູລີ"},
            {8, "ຊຽງຂວາງ"},
            {9, "ວຽງຈັນ"},
            {10, "ບໍລິຄຳໄຊ"},
            {11, "ຄຳມ່ວນ"},
            {12, "ສະຫວັນນະເຂດ"},
            {13, "ສາລະວັນ"},
            {14, "ເຊກອງ"},
            {15, "ຈຳປາສັກ"},
            {16, "ອັດຕະປື"},
            {17, "ໄຊສົມບູນ"}
        };

        public LaoProvinceDetector1Stage(string onnxPath)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Detects and classifies the Lao province directly from a 320x160 plate.
        /// </summary>
        /// <param name="rectifiedPlate">320x160 BGR plate image</param>
        /// <param name="confThreshold">Confidence threshold (default: 0.15)</param>
        /// <returns>Tuple of (Province Name, Confidence Percentage, Banner Box) or null if not detected</returns>
        public (string Province, float Conf, Rect BannerBox)? DetectProvince(Mat rectifiedPlate, float confThreshold = 0.15f)
        {
            int origW = rectifiedPlate.Width;  // 320
            int origH = rectifiedPlate.Height; // 160

            // 1. Direct Stretch Resize to 320x320 RGB and normalize
            using var rgb320 = new Mat();
            Cv2.Resize(rectifiedPlate, rgb320, new Size(320, 320), 0, 0, InterpolationFlags.Linear);
            Cv2.CvtColor(rgb320, rgb320, ColorConversionCodes.BGR2RGB);

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

            // 2. Run ONNX Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>(); // [1, 2125, 22]

            int numCandidates = output.Dimensions[1];
            float bestScore = -1f;
            int bestCls = -1;
            Rect bestBox = default;

            float scaleX = origW / 320.0f;
            float scaleY = origH / 320.0f;

            for (int i = 0; i < numCandidates; i++)
            {
                // Classes 0..17 start from index 4 to 21
                float maxClsScore = -1f;
                int currentCls = -1;
                for (int c = 0; c < 18; c++)
                {
                    float clsProb = output[0, i, 4 + c];
                    if (clsProb > maxClsScore)
                    {
                        maxClsScore = clsProb;
                        currentCls = c;
                    }
                }

                if (maxClsScore > confThreshold && maxClsScore > bestScore)
                {
                    bestScore = maxClsScore;
                    bestCls = currentCls;

                    int x1 = Math.Max(0, (int)Math.Round(output[0, i, 0] * scaleX));
                    int y1 = Math.Max(0, (int)Math.Round(output[0, i, 1] * scaleY));
                    int x2 = Math.Min(origW, (int)Math.Round(output[0, i, 2] * scaleX));
                    int y2 = Math.Min(origH, (int)Math.Round(output[0, i, 3] * scaleY));

                    bestBox = new Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1));
                }
            }

            if (bestCls >= 0)
            {
                return (ProvinceMap[bestCls], bestScore * 100f, bestBox);
            }
            return null;
        }

        public void Dispose() => _session?.Dispose();
    }
}
