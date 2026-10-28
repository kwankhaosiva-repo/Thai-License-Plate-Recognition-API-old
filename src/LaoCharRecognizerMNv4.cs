using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLpr.Lao
{
    /// <summary>
    /// Model 3B: Lao Character Recognizer (MobileNetV4 Lao v7 ONNX, 224x224, 30 Classes)
    /// Recognizes Lao consonants and digits (0..9: digits, 10..29: Lao consonants).
    ///
    /// Preprocessing Contract (Serve == Train):
    /// 1. Crop character bounding box with +2px cushion margin.
    /// 2. Compute ambient background color from 4 corners of the crop.
    /// 3. Pad to square (maxDim x maxDim) with ambient background color.
    /// 4. Autocontrast (cutoff=2%) matching PIL ImageOps.autocontrast.
    /// 5. Resize to 224x224 RGB using Bicubic interpolation.
    /// 6. ImageNet Normalization (0-255 scale):
    ///    Mean: [123.675, 116.28, 103.53], Std: [58.395, 57.12, 57.375]
    /// 7. Grammar Masking:
    ///    - First 2 characters (index 0, 1): Lao consonants only (classes 10..29)
    ///    - Subsequent characters (index 2..): Numeric digits only (classes 0..9)
    /// </summary>
    public class LaoCharRecognizerMNv4 : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string _inputName;

        // ImageNet normalization constants in 0-255 scale (RGB order)
        private static readonly float[] Mean = { 123.675f, 116.28f, 103.53f };
        private static readonly float[] Std  = { 58.395f, 57.12f, 57.375f };

        // v7 Class Mapping (30 classes matching char_classifier_map_lao_v7.json)
        public static readonly Dictionary<int, string> CharMap = new()
        {
            { 0, "0" }, { 1, "1" }, { 2, "2" }, { 3, "3" }, { 4, "4" },
            { 5, "5" }, { 6, "6" }, { 7, "7" }, { 8, "8" }, { 9, "9" },
            { 10, "ກ" }, { 11, "ຂ" }, { 12, "ຄ" }, { 13, "ຈ" }, { 14, "ຍ" },
            { 15, "ດ" }, { 16, "ຕ" }, { 17, "ທ" }, { 18, "ນ" }, { 19, "ບ" },
            { 20, "ຜ" }, { 21, "ພ" }, { 22, "ມ" }, { 23, "ຣ" }, { 24, "ລ" },
            { 25, "ວ" }, { 26, "ສ" }, { 27, "ຫ" }, { 28, "ອ" }, { 29, "ຮ" }
        };

        public class RecognitionResult
        {
            public string Character { get; set; } = string.Empty;
            public int ClassId { get; set; }
            public float Probability { get; set; }
            public Rect Box { get; set; }
        }

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
        /// <param name="charIndex">Zero-based sequence index on plate (0..1 for consonants, 2+ for digits). Pass -1 for auto-spatial heuristic.</param>
        /// <returns>Tuple of (Recognized Character Symbol, Confidence Percentage)</returns>
        public (string Char, float Prob) RecognizeChar(Mat rectifiedPlate, Rect box, int charIndex = -1)
        {
            var detail = RecognizeCharDetailed(rectifiedPlate, box, charIndex);
            return (detail.Character, detail.Probability);
        }

        /// <summary>
        /// Recognizes an entire license plate from a list of detected character boxes (sorted left-to-right).
        /// Applies sequential grammar (first 2 consonants, trailing digits) and formats text e.g. "ກກ 1234".
        /// </summary>
        public (string FormattedPlate, float AvgConf, List<RecognitionResult> Details) RecognizePlate(Mat rectifiedPlate, List<Rect> charBoxes)
        {
            var details = new List<RecognitionResult>();
            if (charBoxes == null || charBoxes.Count == 0)
                return (string.Empty, 0.0f, details);

            var sortedBoxes = charBoxes.OrderBy(b => b.X).ToList();
            var charStrings = new List<string>();
            float totalConf = 0.0f;

            for (int i = 0; i < sortedBoxes.Count; i++)
            {
                var res = RecognizeCharDetailed(rectifiedPlate, sortedBoxes[i], charIndex: i);
                details.Add(res);
                charStrings.Add(res.Character);
                totalConf += res.Probability;
            }

            float avgConf = totalConf / sortedBoxes.Count;

            // Format plate text: If >= 2 characters, separate consonants and digits with space: "ກກ 1234"
            string formatted;
            if (charStrings.Count > 2)
            {
                string consonants = string.Concat(charStrings.Take(2));
                string digits = string.Concat(charStrings.Skip(2));
                formatted = $"{consonants} {digits}";
            }
            else
            {
                formatted = string.Concat(charStrings);
            }

            return (formatted, avgConf, details);
        }

        /// <summary>
        /// Detailed single character recognition with grammar enforcement and confidence scores.
        /// </summary>
        public RecognitionResult RecognizeCharDetailed(Mat rectifiedPlate, Rect box, int charIndex = -1)
        {
            int rw = rectifiedPlate.Width;
            int rh = rectifiedPlate.Height;

            // 1. Adaptive margin cushion (+2px) to preserve letter baselines and ascenders/descenders
            int cx1 = Math.Max(0, box.X - 2);
            int cx2 = Math.Min(rw, box.Right + 2);
            int cy1 = Math.Max(0, box.Y - 1);
            int cy2 = Math.Min((int)(rh * 0.94f), box.Bottom + 2);

            int cw = Math.Max(1, cx2 - cx1);
            int ch = Math.Max(1, cy2 - cy1);
            using var charCrop = new Mat(rectifiedPlate, new Rect(cx1, cy1, cw, ch));

            // 2. Ambient background padding (Median of 4 corners)
            Vec3b[] corners = new Vec3b[]
            {
                charCrop.At<Vec3b>(0, 0),
                charCrop.At<Vec3b>(0, charCrop.Width - 1),
                charCrop.At<Vec3b>(charCrop.Height - 1, 0),
                charCrop.At<Vec3b>(charCrop.Height - 1, charCrop.Width - 1)
            };
            byte bgB = (byte)((corners[1].Item0 + corners[2].Item0) / 2);
            byte bgG = (byte)((corners[1].Item1 + corners[2].Item1) / 2);
            byte bgR = (byte)((corners[1].Item2 + corners[2].Item2) / 2);

            int maxDim = Math.Max(charCrop.Width, charCrop.Height);
            using var padded = new Mat(maxDim, maxDim, MatType.CV_8UC3, new Scalar(bgB, bgG, bgR));
            int offsetX = (maxDim - charCrop.Width) / 2;
            int offsetY = (maxDim - charCrop.Height) / 2;
            using var padRoi = new Mat(padded, new Rect(offsetX, offsetY, charCrop.Width, charCrop.Height));
            charCrop.CopyTo(padRoi);

            // 3. Autocontrast (cutoff=2%) matching PIL ImageOps.autocontrast(cutoff=2)
            using var contrastImg = ApplyAutocontrast(padded, cutoffPercent: 2);

            // 4. Resize to 224x224 RGB and ImageNet normalize
            using var rgb224 = new Mat();
            Cv2.Resize(contrastImg, rgb224, new Size(224, 224), 0, 0, InterpolationFlags.Cubic);
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

            // 5. Run ONNX Inference
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
            using var results = _session.Run(inputs);
            float[] logits = results.First().AsTensor<float>().ToArray(); // 30 classes

            // 6. Sequence & Grammar Masking:
            // Lao license plates: Exactly 2 consonants (classes 10..29), followed by digits (classes 0..9).
            bool isConsonant;
            if (charIndex >= 0)
            {
                isConsonant = (charIndex < 2);
            }
            else
            {
                // Fallback to spatial position: First 38% width = consonants
                float centerX = box.X + (box.Width / 2.0f);
                isConsonant = centerX < (0.38f * rw);
            }

            if (isConsonant)
            {
                // Mask digits (0..9)
                for (int i = 0; i < 10; i++) logits[i] = -1e9f;
            }
            else
            {
                // Mask consonants (10..29)
                for (int i = 10; i < logits.Length; i++) logits[i] = -1e9f;
            }

            // 7. Numerically-stable Softmax
            float maxL = logits.Max();
            float[] exp = logits.Select(l => (float)Math.Exp(l - maxL)).ToArray();
            float sumExp = exp.Sum();
            float[] probs = exp.Select(e => e / sumExp).ToArray();

            int bestIdx = 0;
            float bestProb = probs[0];
            for (int i = 1; i < probs.Length; i++)
            {
                if (probs[i] > bestProb)
                {
                    bestProb = probs[i];
                    bestIdx = i;
                }
            }

            return new RecognitionResult
            {
                Character = CharMap.TryGetValue(bestIdx, out var sym) ? sym : "?",
                ClassId = bestIdx,
                Probability = bestProb * 100.0f,
                Box = box
            };
        }

        /// <summary>
        /// Matches PIL ImageOps.autocontrast(cutoff=2) in OpenCV C#
        /// </summary>
        private static Mat ApplyAutocontrast(Mat bgr, int cutoffPercent = 2)
        {
            var channels = Cv2.Split(bgr);
            var resultChannels = new Mat[channels.Length];

            for (int c = 0; c < channels.Length; c++)
            {
                using var hist = new Mat();
                Cv2.CalcHist(new[] { channels[c] }, new[] { 0 }, null, hist, 1, new[] { 256 }, new[] { new Rangef(0, 256) });

                int total = channels[c].Rows * channels[c].Cols;
                int cutCount = (total * cutoffPercent) / 100;

                int low = 0;
                int count = 0;
                for (int i = 0; i < 256; i++)
                {
                    count += (int)hist.At<float>(i);
                    if (count > cutCount)
                    {
                        low = i;
                        break;
                    }
                }

                int high = 255;
                count = 0;
                for (int i = 255; i >= 0; i--)
                {
                    count += (int)hist.At<float>(i);
                    if (count > cutCount)
                    {
                        high = i;
                        break;
                    }
                }

                resultChannels[c] = new Mat();
                if (high > low)
                {
                    double alpha = 255.0 / (high - low);
                    double beta = -low * alpha;
                    channels[c].ConvertTo(resultChannels[c], MatType.CV_8UC1, alpha, beta);
                }
                else
                {
                    channels[c].CopyTo(resultChannels[c]);
                }
                channels[c].Dispose();
            }

            var merged = new Mat();
            Cv2.Merge(resultChannels, merged);
            foreach (var ch in resultChannels) ch.Dispose();
            return merged;
        }

        public void Dispose() => _session?.Dispose();
    }
}
