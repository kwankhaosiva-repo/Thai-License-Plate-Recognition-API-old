using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

public class PlateCornerRegressor : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;

    // ImageNet standard normalization constants
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    public PlateCornerRegressor(string onnxModelPath)
    {
        var options = new SessionOptions();
        options.AppendExecutionProvider_CPU(0); // Or CUDA if GPU available
        _session = new InferenceSession(onnxModelPath, options);
        _inputName = _session.InputMetadata.Keys.First();
    }

    /// <summary>
    /// Predicts the 4 corner points in full-image pixel coordinates from an AABB plate detection.
    /// </summary>
    /// <param name="fullImage">Full camera frame (BGR Mat)</param>
    /// <param name="plateBbox">AABB Bounding Box from Model 1 (e.g. PicoDet)</param>
    /// <returns>4 Point2f array: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]</returns>
    public Point2f[] PredictCorners(Mat fullImage, Rect plateBbox)
    {
        int imgW = fullImage.Width;
        int imgH = fullImage.Height;

        // 1. Apply 12% padding around the bounding box (same as Python API server)
        int padW = (int)(plateBbox.Width * 0.12f);
        int padH = (int)(plateBbox.Height * 0.12f);

        int cX1 = Math.Max(0, plateBbox.X - padW);
        int cY1 = Math.Max(0, plateBbox.Y - padH);
        int cX2 = Math.Min(imgW, plateBbox.Right + padW);
        int cY2 = Math.Min(imgH, plateBbox.Bottom + padH);

        int cropW = Math.Max(1, cX2 - cX1);
        int cropH = Math.Max(1, cY2 - cY1);

        using var cropBgr = new Mat(fullImage, new Rect(cX1, cY1, cropW, cropH));
        using var resizedCrop = new Mat();
        Cv2.Resize(cropBgr, resizedCrop, new Size(224, 224), 0, 0, InterpolationFlags.Linear);

        // 2. Preprocess: BGR -> RGB and Normalize (ImageNet mean & std)
        var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                Vec3b pixel = resizedCrop.At<Vec3b>(y, x); // BGR
                float r = (pixel.Item2 / 255.0f - Mean[0]) / Std[0];
                float g = (pixel.Item1 / 255.0f - Mean[1]) / Std[1];
                float b = (pixel.Item0 / 255.0f - Mean[2]) / Std[2];

                tensor[0, 0, y, x] = r;
                tensor[0, 1, y, x] = g;
                tensor[0, 2, y, x] = b;
            }
        }

        // 3. Run ONNX Inference
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
        using var results = _session.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Output contains 8 normalized floats: [tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y]
        float[] normCoords = outputTensor.ToArray();

        // 4. Map normalized local crop coordinates back to full image pixel coordinates
        var corners = new Point2f[4];
        for (int i = 0; i < 4; i++)
        {
            float localX = normCoords[i * 2];
            float localY = normCoords[i * 2 + 1];

            corners[i] = new Point2f(
                cX1 + (localX * cropW),
                cY1 + (localY * cropH)
            );
        }

        return OrderQuadClockwise(corners);
    }

    /// <summary>
    /// Step 1 & 2: Perspective Warps and deskews the tilted plate into a flat 320x160 image.
    /// </summary>
    public Mat WarpPerspectivePlate(Mat fullImage, Point2f[] quadCorners, int targetW = 320, int targetH = 160)
    {
        Point2f[] dstPoints = new Point2f[]
        {
            new Point2f(0, 0),
            new Point2f(targetW - 1, 0),
            new Point2f(targetW - 1, targetH - 1),
            new Point2f(0, targetH - 1)
        };

        using var matrix = Cv2.GetPerspectiveTransform(quadCorners, dstPoints);
        var warped = new Mat();
        Cv2.WarpPerspective(fullImage, warped, matrix, new Size(targetW, targetH), InterpolationFlags.Cubic);
        return FineDeskewPlate(warped);
    }

    /// <summary>
    /// Step 2: Fine-tunes residual tilt (±1° to 5°) using HoughLines horizontal line angle.
    /// Matches Python fine_deskew_plate() exactly.
    /// </summary>
    public Mat FineDeskewPlate(Mat plateBgr)
    {
        using var gray = new Mat();
        Cv2.CvtColor(plateBgr, gray, ColorConversionCodes.BGR2GRAY);

        using var edges = new Mat();
        Cv2.Canny(gray, edges, 50, 150);

        // Find straight lines with probabilistic Hough transform
        LineSegmentPoint[] lines = Cv2.HoughLinesP(edges, 1, Math.PI / 180.0, threshold: 40, minLineLength: 30, maxLineGap: 10);
        if (lines == null || lines.Length == 0)
            return plateBgr;

        var angles = new List<double>();
        foreach (var line in lines)
        {
            double dx = line.P2.X - line.P1.X;
            double dy = line.P2.Y - line.P1.Y;
            double angleDeg = Math.Atan2(dy, dx) * (180.0 / Math.PI);

            // Only consider near-horizontal lines (plate frame or text baseline)
            if (Math.Abs(angleDeg) < 15.0)
            {
                angles.Add(angleDeg);
            }
        }

        if (angles.Count == 0)
            return plateBgr;

        // Calculate median angle
        angles.Sort();
        double medianAngle = angles[angles.Count / 2];

        // If residual tilt is negligible (< 0.5°), don't rotate
        if (Math.Abs(medianAngle) < 0.5)
            return plateBgr;

        // Rotate using affine transform
        var center = new Point2f(plateBgr.Width / 2.0f, plateBgr.Height / 2.0f);
        using var rotMatrix = Cv2.GetRotationMatrix2D(center, medianAngle, 1.0);
        var deskewed = new Mat();
        Cv2.WarpAffine(plateBgr, deskewed, rotMatrix, new Size(plateBgr.Width, plateBgr.Height), InterpolationFlags.Linear, BorderTypes.Replicate);

        plateBgr.Dispose();
        return deskewed;
    }

    /// <summary>
    /// Orders 4 quadrilateral 2D points clockwise starting from Top-Left:
    /// [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    /// </summary>
    private Point2f[] OrderQuadClockwise(Point2f[] pts)
    {
        // Sort by sum of (x + y): smallest is Top-Left, largest is Bottom-Right
        var sortedBySum = pts.OrderBy(p => p.X + p.Y).ToArray();
        Point2f tl = sortedBySum[0];
        Point2f br = sortedBySum[3];

        // Sort by difference of (y - x): smallest is Top-Right, largest is Bottom-Left
        var remaining = new[] { sortedBySum[1], sortedBySum[2] };
        Point2f tr = remaining.OrderBy(p => p.Y - p.X).First();
        Point2f bl = remaining.OrderBy(p => p.Y - p.X).Last();

        return new[] { tl, tr, br, bl };
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}
