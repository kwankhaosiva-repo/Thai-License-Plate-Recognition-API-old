using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model 3A: Lao Character Box Detector (PicoDet-S ONNX)
    /// Detects character bounding boxes on Lao license plates.
    /// Expects a rectified plate (320x160), pads it to 3:1 (320x888),
    /// resizes to 320x320 for inference, and maps coordinates back to 320x160.
    /// </summary>
    public class LaoCharDetectorM3A : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        // ImageNet constants in 0-255 scale
        private static readonly float[] Mean = { 123.675f, 116.28f, 103.53f };
        private static readonly float[] Std  = { 58.395f, 57.12f, 57.375f };

        public LaoCharDetectorM3A(string onnxPath)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Detects character boxes on a 320x160 rectified plate image.
        /// </summary>
        /// <param name="rectifiedPlate">320x160 BGR plate image</param>
        /// <param name="confThreshold">Confidence threshold (default: 0.40)</param>
        /// <returns>List of character bounding boxes sorted left-to-right</returns>
        public List<Rect> DetectCharBoxes(Mat rectifiedPlate, float confThreshold = 0.40f)
        {
            int rw = rectifiedPlate.Width;  // 320
            int rh = rectifiedPlate.Height; // 160

            // 1. Pad 3:1 (Top/Bottom 80px, Left/Right 284px) -> total 320 x 888 px
            const int padY = 80;
            const int padX = 284;
            using var platePadded = new Mat();
            Cv2.CopyMakeBorder(rectifiedPlate, platePadded, padY, padY, padX, padX, BorderTypes.Replicate);

            int pw = platePadded.Width;
            int ph = platePadded.Height;

            // 2. Preprocess: Resize to 320x320 RGB and normalize
            using var rgb320 = new Mat();
            Cv2.Resize(platePadded, rgb320, new Size(320, 320), 0, 0, InterpolationFlags.Linear);
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

            // 3. Run ONNX Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>(); // [1, 2125, 5]

            int numBoxes = output.Dimensions[1];
            var rawBoxes = new List<(Rect Box, float Score)>();

            for (int i = 0; i < numBoxes; i++)
            {
                float score = output[0, i, 4];
                if (score < confThreshold) continue;

                float x1 = output[0, i, 0] * (pw / 320.0f);
                float y1 = output[0, i, 1] * (ph / 320.0f);
                float x2 = output[0, i, 2] * (pw / 320.0f);
                float y2 = output[0, i, 3] * (ph / 320.0f);

                // Map coordinates back to 320x160 plate (subtract padding)
                int ox1 = Math.Max(0, Math.Min(rw, (int)Math.Round(x1 - padX)));
                int oy1 = Math.Max(0, Math.Min(rh, (int)Math.Round(y1 - padY)));
                int ox2 = Math.Max(0, Math.Min(rw, (int)Math.Round(x2 - padX)));
                int oy2 = Math.Max(0, Math.Min(rh, (int)Math.Round(y2 - padY)));

                int bw = ox2 - ox1;
                int bh = oy2 - oy1;

                // Valid Lao character criteria on 320x160 plates
                if (bw >= 12 && bh >= 45 && oy1 < 105)
                {
                    rawBoxes.Add((new Rect(ox1, oy1, bw, bh), score));
                }
            }

            // 4. Non-Maximum Suppression (IoU 0.28) & sort left-to-right
            return ApplyNms(rawBoxes, 0.28f).OrderBy(r => r.X).ToList();
        }

        private List<Rect> ApplyNms(List<(Rect Box, float Score)> boxes, float iouThreshold)
        {
            var sorted = boxes.OrderByDescending(b => b.Score).Select(b => b.Box).ToList();
            var kept = new List<Rect>();

            foreach (var b in sorted)
            {
                bool overlap = false;
                foreach (var kb in kept)
                {
                    int iw = Math.Max(0, Math.Min(b.Right, kb.Right) - Math.Max(b.Left, kb.Left));
                    int ih = Math.Max(0, Math.Min(b.Bottom, kb.Bottom) - Math.Max(b.Top, kb.Top));
                    float inter = iw * ih;
                    float union = (b.Width * b.Height) + (kb.Width * kb.Height) - inter;
                    if (union > 0 && (inter / union) > iouThreshold)
                    {
                        overlap = true;
                        break;
                    }
                }
                if (!overlap) kept.Add(b);
            }
            return kept;
        }

        public void Dispose() => _session?.Dispose();
    }
}
