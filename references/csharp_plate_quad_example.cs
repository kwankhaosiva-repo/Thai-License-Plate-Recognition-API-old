// ==============================================================================
// C# License Plate 1-Model 4-Corner Polygon Detection & Rectification
// Model: PlateQuadNet (weights/plate_quad_detector_opset18.onnx)
// License: 100% BSD-3 / Apache-2.0 Permissive (Enterprise Commercial Friendly)
// Dependencies: Microsoft.ML.OnnxRuntime, OpenCvSharp4, OpenCvSharp4.Extensions
// ==============================================================================

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PlateRecognition
{
    public class PlateQuadDetector : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _inputSize;
        private readonly float _confThreshold;

        // ImageNet normalization constants
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };

        public PlateQuadDetector(string onnxModelPath, int inputSize = 512, float confThreshold = 0.4f)
        {
            _inputSize = inputSize;
            _confThreshold = confThreshold;

            var sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CPU(); // Or AppendExecutionProvider_CUDA / DirectML
            _session = new InferenceSession(onnxModelPath, sessionOptions);
        }

        /// <summary>
        /// Detects the license plate in a full vehicle photo and unwarps it to a straight rectangular plate.
        /// </summary>
        /// <param name="srcImage">Original full vehicle image (BGR).</param>
        /// <param name="outPlateWidth">Target rectified plate width (default: 320).</param>
        /// <param name="outPlateHeight">Target rectified plate height (default: 160).</param>
        /// <returns>Straightened plate Mat, or null if no plate detected.</returns>
        public Mat DetectAndRectify(Mat srcImage, int outPlateWidth = 320, int outPlateHeight = 160)
        {
            int origW = srcImage.Width;
            int origH = srcImage.Height;

            // 1. Preprocess: Resize to 512x512, BGR -> RGB, Normalize -> Tensor (1, 3, 512, 512)
            using var resized = new Mat();
            Cv2.Resize(srcImage, resized, new Size(_inputSize, _inputSize));

            using var rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

            var tensor = new DenseTensor<float>(new[] { 1, 3, _inputSize, _inputSize });

            // Fill tensor NCHW
            for (int y = 0; y < _inputSize; y++)
            {
                for (int x = 0; x < _inputSize; x++)
                {
                    Vec3b pixel = rgb.At<Vec3b>(y, x);
                    tensor[0, 0, y, x] = ((pixel.Item0 / 255.0f) - Mean[0]) / Std[0]; // R
                    tensor[0, 1, y, x] = ((pixel.Item1 / 255.0f) - Mean[1]) / Std[1]; // G
                    tensor[0, 2, y, x] = ((pixel.Item2 / 255.0f) - Mean[2]) / Std[2]; // B
                }
            }

            // 2. Run 1-pass Inference
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", tensor)
            };

            using var results = _session.Run(inputs);

            // Output 0: "corners" -> Shape: (1, TopK, 4, 2)
            // Output 1: "scores"  -> Shape: (1, TopK)
            var cornersTensor = results.First(r => r.Name == "corners").AsTensor<float>();
            var scoresTensor  = results.First(r => r.Name == "scores").AsTensor<float>();

            // Inspect top-1 detection
            float topScore = scoresTensor[0, 0];
            if (topScore < _confThreshold)
            {
                Console.WriteLine($"[PlateQuad] No plate detected with confidence >= {_confThreshold} (Best: {topScore:F2})");
                return null;
            }

            Console.WriteLine($"[PlateQuad] Plate detected! Confidence: {topScore * 100:F1}%");

            // 3. Extract 4 Corners: Normalized [0, 1] -> Original Image Pixels
            // Order: [0]=Top-Left, [1]=Top-Right, [2]=Bottom-Right, [3]=Bottom-Left
            var srcQuad = new Point2f[4];
            for (int i = 0; i < 4; i++)
            {
                float normX = cornersTensor[0, 0, i, 0];
                float normY = cornersTensor[0, 0, i, 1];
                srcQuad[i] = new Point2f(normX * origW, normY * origH);
            }

            // Destination straight rectangle
            var dstQuad = new Point2f[4]
            {
                new Point2f(0, 0),
                new Point2f(outPlateWidth, 0),
                new Point2f(outPlateWidth, outPlateHeight),
                new Point2f(0, outPlateHeight)
            };

            // 4. Perspective Transform (Unwarp tilted plate to straight)
            using var transformMatrix = Cv2.GetPerspectiveTransform(srcQuad, dstQuad);
            var rectifiedPlate = new Mat();
            Cv2.WarpPerspective(srcImage, rectifiedPlate, transformMatrix, new Size(outPlateWidth, outPlateHeight));

            return rectifiedPlate;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            string modelPath = "weights/plate_quad_detector_opset18.onnx";
            string testImagePath = "test_car.jpg";

            using var detector = new PlateQuadDetector(modelPath, inputSize: 512, confThreshold: 0.4f);
            using var src = Cv2.ImRead(testImagePath);

            if (src.Empty())
            {
                Console.WriteLine("Failed to load test image.");
                return;
            }

            using var straightPlate = detector.DetectAndRectify(src, outPlateWidth: 320, outPlateHeight: 160);
            if (straightPlate != null)
            {
                Cv2.ImWrite("straight_plate_output.jpg", straightPlate);
                Console.WriteLine("Straightened license plate saved to straight_plate_output.jpg!");
            }
        }
    }
}
