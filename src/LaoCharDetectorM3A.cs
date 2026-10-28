using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model 3A: Lao Character Box Detector (PicoDet-S Lao v6 ONNX, 320x320)
    /// Detects individual character bounding boxes on Lao license plates.
    ///
    /// Preprocessing Contract (Serve == Train):
    /// 1. Input plate: Rectified 320x160 BGR image.
    /// 2. Letterbox/Padding to ~3:1 aspect ratio:
    ///    padY = 80 (top/bottom) -> height becomes 320px
    ///    padX = 284 (left/right) -> width becomes 888px
    ///    cv2.copyMakeBorder(..., BORDER_REPLICATE)
    /// 3. Resize padded plate to 320x320 RGB (Linear Interpolation).
    /// 4. ImageNet Normalization (0-255 scale):
    ///    Mean: [123.675, 116.28, 103.53], Std: [58.395, 57.12, 57.375]
    /// 5. Output shape: [1, 2125, 5] -> [x1, y1, x2, y2, score] (Single class: 'char')
    /// 6. Post-processing: Map coordinates back to 320x160 (subtract padX, padY),
    ///    filter valid characters (width >= 12, height >= 45, y1 < 105),
    ///    apply IoU NMS (0.28) and sort left-to-right.
    /// </summary>
    public class LaoCharDetectorM3A : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        // ImageNet constants in 0-255 scale (RGB order)
        private static readonly float[] Mean = { 123.675f, 116.28f, 103.53f };
        private static readonly float[] Std  = { 58.395f, 57.12f, 57.375f };

        public class DetectedCharBox
        {
            public Rect Box { get; set; }
            public float Score { get; set; }
        }

        public LaoCharDetectorM3A(string onnxPath)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxPath, options);
            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Detects character boxes on a 320x160 rectified plate image.
        /// Returns character bounding boxes sorted left-to-right.
        /// </summary>
        /// <param name="rectifiedPlate">320x160 BGR plate image</param>
        /// <param name="confThreshold">Confidence threshold (default: 0.35)</param>
        /// <returns>List of character bounding boxes sorted left-to-right</returns>
        public List<Rect> DetectCharBoxes(Mat rectifiedPlate, float confThreshold = 0.35f)
        {
            var detailed = DetectCharBoxesDetailed(rectifiedPlate, confThreshold);
            return detailed.Select(d => d.Box).ToList();
        }

        /// <summary>
        /// Detects character boxes with individual confidence scores, sorted left-to-right.
        /// </summary>
        /// <param name="rectifiedPlate">320x160 BGR plate image</param>
        /// <param name="confThreshold">Confidence threshold (default: 0.35)</param>
        /// <param name="iouThreshold">NMS IoU threshold (default: 0.28)</param>
        /// <returns>List of detected boxes with scores sorted left-to-right</returns>
        public List<DetectedCharBox> DetectCharBoxesDetailed(Mat rectifiedPlate, float confThreshold = 0.35f, float iouThreshold = 0.28f)
        {
            if (rectifiedPlate == null || rectifiedPlate.Empty() || rectifiedPlate.Width < 20 || rectifiedPlate.Height < 20)
                return new List<DetectedCharBox>();

            int rw = rectifiedPlate.Width;  // typically 320
            int rh = rectifiedPlate.Height; // typically 160

            // 1. Pad 3:1 (Top/Bottom 80px, Left/Right 284px) -> total 320 x (rw + 568) px
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
            var rawBoxes = new List<DetectedCharBox>();

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
                    rawBoxes.Add(new DetectedCharBox
                    {
                        Box = new Rect(ox1, oy1, bw, bh),
                        Score = score
                    });
                }
            }

            // 4. Non-Maximum Suppression (IoU 0.28) & sort left-to-right
            return ApplyNms(rawBoxes, iouThreshold).OrderBy(r => r.Box.X).ToList();
        }

        private static List<DetectedCharBox> ApplyNms(List<DetectedCharBox> boxes, float iouThreshold)
        {
            var sorted = boxes.OrderByDescending(b => b.Score).ToList();
            var kept = new List<DetectedCharBox>();

            foreach (var b in sorted)
            {
                bool overlap = false;
                foreach (var kb in kept)
                {
                    int iw = Math.Max(0, Math.Min(b.Box.Right, kb.Box.Right) - Math.Max(b.Box.Left, kb.Box.Left));
                    int ih = Math.Max(0, Math.Min(b.Box.Bottom, kb.Box.Bottom) - Math.Max(b.Box.Top, kb.Box.Top));
                    float inter = iw * ih;
                    float union = (b.Box.Width * b.Box.Height) + (kb.Box.Width * kb.Box.Height) - inter;
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
