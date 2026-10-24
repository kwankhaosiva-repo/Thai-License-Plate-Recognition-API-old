using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model: Lao Province Classifier 2-Stage (ResNet18 Grayscale Lao v6 ONNX, 256x64)
    /// High-Resolution Province Classifier: reads the cropped province banner patch,
    /// converts to Grayscale, resizes to 256x64, duplicates across 3 channels with
    /// (val/255.0 - 0.5) / 0.5 normalization, and outputs the 18 Lao province probabilities.
    /// Delivers 96-97% accuracy on challenging/blurry images.
    /// </summary>
    public class LaoProvinceClassifier2Stage : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        public LaoProvinceClassifier2Stage(string onnxPath)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Classifies the province from a cropped banner image.
        /// </summary>
        /// <param name="provinceCrop">Cropped banner image (BGR or Grayscale)</param>
        /// <returns>Tuple of (Province Name, Confidence Percentage)</returns>
        public (string Province, float Conf) ClassifyProvince(Mat provinceCrop)
        {
            if (provinceCrop == null || provinceCrop.Empty() || provinceCrop.Width < 4 || provinceCrop.Height < 4)
            {
                return ("Unknown", 0.0f);
            }

            // 1. Convert to Grayscale
            using var gray = new Mat();
            if (provinceCrop.Channels() == 3)
            {
                Cv2.CvtColor(provinceCrop, gray, ColorConversionCodes.BGR2GRAY);
            }
            else
            {
                provinceCrop.CopyTo(gray);
            }

            // 2. Resize to 256x64 (Width: 256, Height: 64)
            using var resizedGray = new Mat();
            Cv2.Resize(gray, resizedGray, new Size(256, 64), 0, 0, InterpolationFlags.Bilinear);

            // 3. Preprocess Tensor: 3 channels with Grayscale duplicated
            // Normalization: (val / 255.0 - 0.5) / 0.5 -> maps [0, 255] to [-1.0, 1.0]
            var tensor = new DenseTensor<float>(new[] { 1, 3, 64, 256 });
            for (int y = 0; y < 64; y++)
            {
                for (int x = 0; x < 256; x++)
                {
                    byte val = resizedGray.At<byte>(y, x);
                    float normVal = (val / 255.0f - 0.5f) / 0.5f;

                    tensor[0, 0, y, x] = normVal; // Channel 0
                    tensor[0, 1, y, x] = normVal; // Channel 1
                    tensor[0, 2, y, x] = normVal; // Channel 2
                }
            }

            // 4. Run ONNX Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            float[] logits = results.First().AsTensor<float>().ToArray(); // 18 classes

            // 5. Softmax to calculate confidence percentage
            float maxL = logits.Max();
            float[] exp = logits.Select(l => (float)Math.Exp(l - maxL)).ToArray();
            float sumExp = exp.Sum();
            float[] probs = exp.Select(e => e / sumExp).ToArray();

            int topIdx = Array.IndexOf(probs, probs.Max());
            string provName = LaoProvinceDetector1Stage.ProvinceMap.TryGetValue(topIdx, out var name)
                ? name
                : $"Class_{topIdx}";

            return (provName, probs[topIdx] * 100f);
        }

        public void Dispose() => _session?.Dispose();
    }
}
