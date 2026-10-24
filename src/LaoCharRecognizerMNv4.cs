using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model 3B: Lao Character Recognizer (MobileNetV4 Lao v7 ONNX)
    /// Recognizes Lao consonants and digits (30 classes total: 0..9 digits, 10..29 consonants).
    /// Applies square padding, cubic resize to 224x224 RGB, ImageNet normalization,
    /// and spatial grammar masking (consonants on left 38%, digits on right 62%).
    /// </summary>
    public class LaoCharRecognizerMNv4 : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        // ImageNet normalization constants in 0-255 scale
        private static readonly float[] Mean = { 123.675f, 116.28f, 103.53f };
        private static readonly float[] Std  = { 58.395f, 57.12f, 57.375f };

        // v7 Class Mapping (30 classes)
        public static readonly Dictionary<int, string> CharMap = new()
        {
            {0, "0"}, {1, "1"}, {2, "2"}, {3, "3"}, {4, "4"},
            {5, "5"}, {6, "6"}, {7, "7"}, {8, "8"}, {9, "9"},
            {10, "ກ"}, {11, "ຂ"}, {12, "ຄ"}, {13, "ງ"}, {14, "ຈ"},
            {15, "ສ"}, {16, "ຊ"}, {17, "ຍ"}, {18, "ດ"}, {19, "ຕ"},
            {20, "ຖ"}, {21, "ທ"}, {22, "ນ"}, {23, "ບ"}, {24, "ປ"},
            {25, "ຜ"}, {26, "ຝ"}, {27, "ພ"}, {28, "ຟ"}, {29, "ມ"}
        };

        public LaoCharRecognizerMNv4(string onnxPath)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Recognizes a single character given its bounding box on a 320x160 plate.
        /// </summary>
        /// <param name="rectifiedPlate">320x160 BGR plate image</param>
        /// <param name="box">Bounding box of the character</param>
        /// <returns>Tuple of (Recognized Character Symbol, Confidence Percentage)</returns>
        public (string Char, float Prob) RecognizeChar(Mat rectifiedPlate, Rect box)
        {
            using var charCrop = new Mat(rectifiedPlate, box);

            // 1. Square padding (preserves character aspect ratio)
            int maxDim = Math.Max(charCrop.Width, charCrop.Height);
            using var padded = new Mat(maxDim, maxDim, MatType.CV_8UC3, new Scalar(255, 255, 255));
            int offsetX = (maxDim - charCrop.Width) / 2;
            int offsetY = (maxDim - charCrop.Height) / 2;
            charCrop.CopyTo(new Mat(padded, new Rect(offsetX, offsetY, charCrop.Width, charCrop.Height)));

            // 2. Resize to 224x224 RGB and normalize
            using var rgb224 = new Mat();
            Cv2.Resize(padded, rgb224, new Size(224, 224), 0, 0, InterpolationFlags.Cubic);
            Cv2.CvtColor(rgb224, rgb224, ColorConversionCodes.BGR2RGB);

            var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            for (int y = 0; y < 224; y++)
            {
                for (int x = 0; x < 224; x++)
                {
                    Vec3b p = rgb224.At<Vec3b>(y, x);
                    tensor[0, 0, y, x] = (p.Item0 - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = (p.Item1 - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = (p.Item2 - Mean[2]) / Std[2];
                }
            }

            // 3. Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            float[] logits = results.First().AsTensor<float>().ToArray();

            // 4. Spatial Grammar Masking:
            // First 38% width -> Consonants only (indices >= 10)
            // Remaining 62% width -> Digits only (indices 0..9)
            float centerX = box.X + (box.Width / 2.0f);
            bool isConsonantZone = centerX < (0.38f * rectifiedPlate.Width);

            if (isConsonantZone)
            {
                for (int i = 0; i < 10; i++) logits[i] = -1e9f;
            }
            else
            {
                for (int i = 10; i < logits.Length; i++) logits[i] = -1e9f;
            }

            // 5. Softmax
            float maxL = logits.Max();
            float[] exp = logits.Select(l => (float)Math.Exp(l - maxL)).ToArray();
            float sumExp = exp.Sum();
            float[] probs = exp.Select(e => e / sumExp).ToArray();

            int bestIdx = Array.IndexOf(probs, probs.Max());
            return (CharMap[bestIdx], probs[bestIdx] * 100f);
        }

        public void Dispose() => _session?.Dispose();
    }
}
