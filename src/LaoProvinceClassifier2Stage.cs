using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model: Lao Province Classifier 2-Stage (ResNet18 Grayscale Lao v6 ONNX, 256x64, 19 Classes)
    /// High-Resolution Province Classifier: reads the cropped province banner patch,
    /// applies GrayscaleSmartResize (Aspect 4:1 with ambient corner padding),
    /// duplicates across 3 channels with (val/255.0 - 0.5) / 0.25 normalization,
    /// and outputs probabilities for all 19 Lao province classes.
    ///
    /// Preprocessing Contract (Serve == Train):
    /// 1. Convert to Grayscale (1-channel).
    /// 2. Sample 4 corners [ (0,0), (0,W-1), (H-1,0), (H-1,W-1) ] -> Compute median ambient background byte.
    /// 3. Aspect-preserving resize: ratio = min(256.0/W, 64.0/H), newW = max(4, round(W*ratio)), newH = max(4, round(H*ratio)).
    ///    Bicubic interpolation (InterpolationFlags.Cubic).
    /// 4. Center-paste into 256x64 canvas filled with ambient background byte.
    /// 5. Duplicate across 3 channels (R, G, B) with normalization:
    ///    normVal = (pixelVal / 255.0f - 0.5f) / 0.25f;  // Range: [-2.0, 2.0]
    /// 6. Output tensor shape: [1, 19] raw logits -> Softmax.
    /// </summary>
    public class LaoProvinceClassifier2Stage : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        public const int TargetWidth = 256;
        public const int TargetHeight = 64;

        public LaoProvinceClassifier2Stage(string onnxPath)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Classifies the province from a cropped banner image patch.
        /// </summary>
        /// <param name="provinceCrop">Cropped banner image (BGR or Grayscale)</param>
        /// <returns>Tuple of (Province Name, Confidence Percentage)</returns>
        public (string Province, float Conf) ClassifyProvince(Mat provinceCrop)
        {
            var detail = ClassifyProvinceDetailed(provinceCrop);
            return (detail.Province, detail.Confidence);
        }

        /// <summary>
        /// Classifies the province with detailed class index and top probability breakdown.
        /// </summary>
        public (string Province, int ClassId, float Confidence, float[] Probabilities) ClassifyProvinceDetailed(Mat provinceCrop)
        {
            if (provinceCrop == null || provinceCrop.Empty() || provinceCrop.Width < 2 || provinceCrop.Height < 2)
            {
                return ("Unknown", -1, 0.0f, Array.Empty<float>());
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

            int w = gray.Width;
            int h = gray.Height;

            // 2. Compute Ambient Background Byte from 4 Corners
            byte[] corners = new byte[]
            {
                gray.At<byte>(0, 0),
                gray.At<byte>(0, w - 1),
                gray.At<byte>(h - 1, 0),
                gray.At<byte>(h - 1, w - 1)
            };
            Array.Sort(corners);
            byte bgVal = (byte)((corners[1] + corners[2]) / 2);

            // 3. Aspect-Preserving Smart Resize (4:1 target aspect ratio)
            float ratio = Math.Min((float)TargetWidth / w, (float)TargetHeight / h);
            int newW = Math.Max(4, (int)Math.Round(w * ratio));
            int newH = Math.Max(4, (int)Math.Round(h * ratio));

            using var resized = new Mat();
            Cv2.Resize(gray, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Cubic);

            // 4. Center-Paste onto 256x64 Ambient Background Canvas
            using var canvas = new Mat(new Size(TargetWidth, TargetHeight), MatType.CV_8UC1, new Scalar(bgVal));
            int pasteX = (TargetWidth - newW) / 2;
            int pasteY = (TargetHeight - newH) / 2;
            var roi = new Rect(pasteX, pasteY, newW, newH);
            using var canvasRoi = new Mat(canvas, roi);
            resized.CopyTo(canvasRoi);

            // 5. Preprocess Tensor: [1, 3, 64, 256] with Grayscale duplicated across 3 channels
            // Normalization: (val / 255.0 - 0.5) / 0.25 (matches torchvision mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
            var tensor = new DenseTensor<float>(new[] { 1, 3, TargetHeight, TargetWidth });
            for (int y = 0; y < TargetHeight; y++)
            {
                for (int x = 0; x < TargetWidth; x++)
                {
                    byte val = canvas.At<byte>(y, x);
                    float normVal = (val / 255.0f - 0.5f) / 0.25f;

                    tensor[0, 0, y, x] = normVal; // Channel 0 (R)
                    tensor[0, 1, y, x] = normVal; // Channel 1 (G)
                    tensor[0, 2, y, x] = normVal; // Channel 2 (B)
                }
            }

            // 6. Run ONNX Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            float[] logits = results.First().AsTensor<float>().ToArray(); // 19 classes

            // 7. Numerically stable Softmax to calculate confidence percentage
            float maxL = logits.Max();
            float[] exp = logits.Select(l => (float)Math.Exp(l - maxL)).ToArray();
            float sumExp = exp.Sum();
            float[] probs = exp.Select(e => e / sumExp).ToArray();

            int topIdx = 0;
            float topProb = probs[0];
            for (int i = 1; i < probs.Length; i++)
            {
                if (probs[i] > topProb)
                {
                    topProb = probs[i];
                    topIdx = i;
                }
            }

            string provName = LaoProvinceDetector1Stage.ProvinceMap.TryGetValue(topIdx, out var name)
                ? name
                : $"Class_{topIdx}";

            return (provName, topIdx, topProb * 100.0f, probs);
        }

        public void Dispose() => _session?.Dispose();
    }
}
