// ==============================================================================
// C# Lane Straddling & Departure Detection Engine (BEV Homography)
// Framework: .NET 6.0 / 7.0 / 8.0 (.NET Standard 2.1)
// Dependencies: Microsoft.ML.OnnxRuntime, OpenCvSharp4
// Models: front_wheel_detector_v6.onnx (YOLO11n, MIT), LibreDFINEn.onnx (MIT)
// License: Permissive Apache-2.0 / MIT
// ==============================================================================

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ThaiLPR.LaneSafety
{
    public struct Point2D
    {
        public double X { get; set; }
        public double Y { get; set; }

        public Point2D(double x, double y)
        {
            X = x;
            Y = y;
        }

        public override string ToString() => $"({X:F1}, {Y:F1})";
    }

    public class GroundContact
    {
        public Point2D LeftWheel { get; set; }
        public Point2D RightWheel { get; set; }
        public Point2D CenterBottom { get; set; }
        public Point2D BevLeftWheel { get; set; }
        public Point2D BevRightWheel { get; set; }
        public Point2D BevCenterBottom { get; set; }
        public Rect2f? CabFaceBbox { get; set; }
    }

    public class StraddleAssessment
    {
        public bool IsStraddling { get; set; }
        public string StatusLabel { get; set; } = "IN_LANE";
        public string? ViolationSide { get; set; }
        public double LeftWheelClearanceCm { get; set; }
        public double RightWheelClearanceCm { get; set; }
        public double StraddleDepthCm { get; set; }
        public string Details { get; set; } = string.Empty;
    }

    public class VehicleLaneResult
    {
        public Rect2f Bbox { get; set; }
        public string ClassName { get; set; } = "truck";
        public float Confidence { get; set; }
        public GroundContact GroundContact { get; set; } = new();
        public StraddleAssessment Assessment { get; set; } = new();
    }

    public class RoadCalibration
    {
        public List<Point2D> GroundQuad { get; set; } = new();
        public List<Point2D> LeftLane { get; set; } = new();
        public List<Point2D> RightLane { get; set; } = new();
        public int BevWidth { get; set; } = 400;
        public int BevHeight { get; set; } = 800;
        public int RefWidth { get; set; } = 1280;
        public int RefHeight { get; set; } = 720;
        public double WorldWidthM { get; set; } = 3.5;
        public double WorldLengthM { get; set; } = 35.0;

        public double PixelsPerMeterX => BevWidth / WorldWidthM;
        public double PixelsPerMeterY => BevHeight / WorldLengthM;

        private Mat? _hMat;
        private Mat? _hInvMat;

        public void ComputeHomography()
        {
            if (GroundQuad.Count != 4)
                throw new InvalidOperationException("GroundQuad must contain exactly 4 points.");

            // Auto-square: Level far depth baseline and near depth baseline to eliminate perspective shear
            double yFar = (GroundQuad[0].Y + GroundQuad[1].Y) / 2.0;
            double yNear = (GroundQuad[3].Y + GroundQuad[2].Y) / 2.0;
            GroundQuad[0] = new Point2D(GroundQuad[0].X, yFar);
            GroundQuad[1] = new Point2D(GroundQuad[1].X, yFar);
            GroundQuad[2] = new Point2D(GroundQuad[2].X, yNear);
            GroundQuad[3] = new Point2D(GroundQuad[3].X, yNear);

            if (LeftLane.Count >= 2)
            {
                LeftLane[0] = GroundQuad[0];
                LeftLane[1] = GroundQuad[3];
            }
            if (RightLane.Count >= 2)
            {
                RightLane[0] = GroundQuad[1];
                RightLane[1] = GroundQuad[2];
            }

            var srcPts = GroundQuad.Select(p => new Point2f((float)p.X, (float)p.Y)).ToArray();
            var dstPts = new Point2f[]
            {
                new Point2f(0f, 0f),
                new Point2f(BevWidth - 1, 0f),
                new Point2f(BevWidth - 1, BevHeight - 1),
                new Point2f(0f, BevHeight - 1)
            };

            _hMat?.Dispose();
            _hInvMat?.Dispose();

            _hMat = Cv2.GetPerspectiveTransform(srcPts, dstPts);
            _hInvMat = _hMat.Inv();
        }

        public void AutoSquareCalibration()
        {
            if (GroundQuad.Count != 4)
                throw new InvalidOperationException("GroundQuad must contain exactly 4 points.");

            // Level far depth baseline (G1 Far-Left, G2 Far-Right)
            // Level near depth baseline (G4 Near-Left, G3 Near-Right)
            double yFar = (GroundQuad[0].Y + GroundQuad[1].Y) / 2.0;
            double yNear = (GroundQuad[3].Y + GroundQuad[2].Y) / 2.0;

            GroundQuad[0] = new Point2D(GroundQuad[0].X, yFar);
            GroundQuad[1] = new Point2D(GroundQuad[1].X, yFar);
            GroundQuad[2] = new Point2D(GroundQuad[2].X, yNear);
            GroundQuad[3] = new Point2D(GroundQuad[3].X, yNear);

            LeftLane = new List<Point2D> { GroundQuad[0], GroundQuad[3] };
            RightLane = new List<Point2D> { GroundQuad[1], GroundQuad[2] };

            ComputeHomography();
        }

        public Point2D ProjectToBev(Point2D p)
        {
            if (_hMat == null) ComputeHomography();
            var src = new Point2f[] { new Point2f((float)p.X, (float)p.Y) };
            var dst = Cv2.PerspectiveTransform(src, _hMat!);
            return new Point2D(dst[0].X, dst[0].Y);
        }

        public Point2D ProjectFromBev(Point2D p)
        {
            if (_hInvMat == null) ComputeHomography();
            var src = new Point2f[] { new Point2f((float)p.X, (float)p.Y) };
            var dst = Cv2.PerspectiveTransform(src, _hInvMat!);
            return new Point2D(dst[0].X, dst[0].Y);
        }

        public RoadCalibration RescaleToImage(int imgW, int imgH)
        {
            if (imgW == RefWidth && imgH == RefHeight) return this;
            double sx = (double)imgW / RefWidth;
            double sy = (double)imgH / RefHeight;

            var scaled = new RoadCalibration
            {
                GroundQuad = GroundQuad.Select(p => new Point2D(p.X * sx, p.Y * sy)).ToList(),
                LeftLane = LeftLane.Select(p => new Point2D(p.X * sx, p.Y * sy)).ToList(),
                RightLane = RightLane.Select(p => new Point2D(p.X * sx, p.Y * sy)).ToList(),
                BevWidth = BevWidth,
                BevHeight = BevHeight,
                RefWidth = imgW,
                RefHeight = imgH,
                WorldWidthM = WorldWidthM,
                WorldLengthM = WorldLengthM
            };
            scaled.ComputeHomography();
            return scaled;
        }

        public static RoadCalibration FromJsonFile(string jsonPath, string profile = "1280x720")
        {
            string jsonText = File.ReadAllText(jsonPath);
            using var doc = JsonDocument.Parse(jsonText);
            var root = doc.RootElement;
            var profElem = root.GetProperty("profiles").GetProperty(profile);

            var calib = new RoadCalibration
            {
                BevWidth = profElem.GetProperty("bev_width").GetInt32(),
                BevHeight = profElem.GetProperty("bev_height").GetInt32(),
                RefWidth = profElem.GetProperty("ref_width").GetInt32(),
                RefHeight = profElem.GetProperty("ref_height").GetInt32(),
                WorldWidthM = profElem.GetProperty("world_width_m").GetDouble(),
                WorldLengthM = profElem.GetProperty("world_length_m").GetDouble()
            };

            foreach (var pt in profElem.GetProperty("ground_quad").EnumerateArray())
            {
                calib.GroundQuad.Add(new Point2D(pt[0].GetDouble(), pt[1].GetDouble()));
            }

            var lanesElem = profElem.GetProperty("lanes");
            foreach (var pt in lanesElem.GetProperty("left").EnumerateArray())
                calib.LeftLane.Add(new Point2D(pt[0].GetDouble(), pt[1].GetDouble()));
            foreach (var pt in lanesElem.GetProperty("right").EnumerateArray())
                calib.RightLane.Add(new Point2D(pt[0].GetDouble(), pt[1].GetDouble()));

            calib.ComputeHomography();
            return calib;
        }
    }

    /// <summary>
    /// Production C# Engine for Real-Time Highway Lane Straddling & Departure Detection.
    /// Operates on standalone ONNX models with zero external Python runtime dependencies.
    /// </summary>
    public class LaneStraddlingEngine : IDisposable
    {
        private readonly InferenceSession _wheelSession;
        private readonly RoadCalibration _baseCalibration;
        private const int ModelInputSize = 640;

        // Commercial Heavy Truck steer tire physical half-width is 14cm (Tread width = 28-30cm)
        private const double TruckTireHalfWidthCm = 14.0;
        private const double CarTireHalfWidthCm = 10.0;

        public LaneStraddlingEngine(string wheelOnnxPath, RoadCalibration calibration)
        {
            var options = new SessionOptions();
            options.AppendExecutionProvider_CPU(); // In production, add CUDA or DirectML provider
            _wheelSession = new InferenceSession(wheelOnnxPath, options);
            _baseCalibration = calibration;
        }

        /// <summary>
        /// Process a camera frame and evaluate lane straddling for vehicle bounding boxes.
        /// </summary>
        public List<VehicleLaneResult> ProcessFrame(Mat frameBgr, List<Rect2f> vehicleBboxes, List<string>? vehicleClasses = null)
        {
            int origW = frameBgr.Width;
            int origH = frameBgr.Height;
            var activeCalib = _baseCalibration.RescaleToImage(origW, origH);

            // 1. Run YOLO11 Stage 2 Wheel & Cab Detector on frame
            var detections = RunWheelAndCabDetector(frameBgr);

            var results = new List<VehicleLaneResult>();
            for (int i = 0; i < vehicleBboxes.Count; i++)
            {
                var vBox = vehicleBboxes[i];
                string vClass = (vehicleClasses != null && i < vehicleClasses.Count) ? vehicleClasses[i] : "truck";

                // Match front cab face and front wheel proposals to this vehicle
                var matchedCab = detections.Where(d => d.ClassId == 0 && vBox.Contains(new Point2f(d.Bbox.X + d.Bbox.Width / 2, d.Bbox.Y + d.Bbox.Height / 2)))
                                           .OrderByDescending(d => d.Confidence)
                                           .FirstOrDefault();

                Rect2f? cabBbox = matchedCab?.Bbox;

                // Match front steer wheel proposals inside or near the cab face
                var matchedTires = new List<Detection>();
                foreach (var d in detections.Where(d => d.ClassId == 1 && d.Confidence >= 0.20f))
                {
                    float cx = d.Bbox.X + d.Bbox.Width / 2f;
                    float by2 = d.Bbox.Bottom;
                    if (cabBbox.HasValue)
                    {
                        float marginX = Math.Max(35f, 0.30f * cabBbox.Value.Width);
                        float marginY = Math.Max(45f, 0.35f * cabBbox.Value.Height);
                        if (cx >= cabBbox.Value.Left - marginX && cx <= cabBbox.Value.Right + marginX &&
                            by2 >= cabBbox.Value.Bottom - marginY && by2 <= cabBbox.Value.Bottom + 25f)
                        {
                            matchedTires.Add(d);
                        }
                    }
                    else if (vBox.Contains(new Point2f(cx, by2 - 10f)))
                    {
                        matchedTires.Add(d);
                    }
                }

                // Filter front-most tires (nearest Y to camera) and cluster overlapping proposals
                var frontSteerTires = new List<Detection>();
                if (matchedTires.Count > 0)
                {
                    float maxY = matchedTires.Max(t => t.Bbox.Bottom);
                    var frontProposals = matchedTires.Where(t => t.Bbox.Bottom >= maxY - 20f)
                                                     .OrderByDescending(t => t.Confidence)
                                                     .ToList();

                    // Cluster overlapping proposals on the same physical wheel
                    var clustered = new List<Detection>();
                    foreach (var t in frontProposals)
                    {
                        float tcx = t.Bbox.X + t.Bbox.Width / 2f;
                        float tcy = t.Bbox.Y + t.Bbox.Height / 2f;
                        bool isDup = false;
                        foreach (var c in clustered)
                        {
                            float ccx = c.Bbox.X + c.Bbox.Width / 2f;
                            float ccy = c.Bbox.Y + c.Bbox.Height / 2f;
                            if (Math.Abs(tcx - ccx) < 45f && Math.Abs(tcy - ccy) < 25f)
                            {
                                isDup = true;
                                break;
                            }
                        }
                        if (!isDup) clustered.Add(t);
                    }

                    if (clustered.Count >= 2)
                    {
                        var sorted = clustered.OrderBy(t => t.Bbox.X + t.Bbox.Width / 2f).ToList();
                        var tLeft = sorted.First();
                        var tRight = sorted.Last();
                        float confL = tLeft.Confidence;
                        float confR = tRight.Confidence;
                        float yDiff = Math.Abs(tLeft.Bbox.Bottom - tRight.Bbox.Bottom);

                        // Dominant wheel gating: If one side is visible in LOS while opposite is noisy/occluded
                        if (confL >= 0.40f && confR < 0.35f)
                            frontSteerTires.Add(tLeft);
                        else if (confR >= 0.40f && confL < 0.35f)
                            frontSteerTires.Add(tRight);
                        else if (yDiff <= 18f && Math.Min(confL, confR) >= 0.30f)
                        {
                            frontSteerTires.Add(tLeft);
                            frontSteerTires.Add(tRight);
                        }
                        else
                        {
                            frontSteerTires.Add(confL >= confR ? tLeft : tRight);
                        }
                    }
                    else if (clustered.Count == 1)
                    {
                        frontSteerTires.Add(clustered[0]);
                    }
                }

                // 2. Estimate Physical Ground Contacts with Dynamic Track Width & Ground Y-Constraint
                var groundContact = EstimateVehicleGroundContact(vBox, vClass, frontSteerTires, cabBbox, activeCalib);

                // 3. Assess Lane Straddling with Finite Tire Contact Patch Width
                var assessment = AssessLaneStraddling(groundContact, activeCalib, vClass, toleranceCm: 0.0);

                results.Add(new VehicleLaneResult
                {
                    Bbox = vBox,
                    ClassName = vClass,
                    Confidence = 0.90f,
                    GroundContact = groundContact,
                    Assessment = assessment
                });
            }

            return results;
        }

        private struct Detection
        {
            public int ClassId; // 0 = cab_face, 1 = front_wheel
            public float Confidence;
            public Rect2f Bbox;
        }

        private List<Detection> RunWheelAndCabDetector(Mat frameBgr)
        {
            int origW = frameBgr.Width;
            int origH = frameBgr.Height;

            using var resized = new Mat();
            Cv2.Resize(frameBgr, resized, new Size(ModelInputSize, ModelInputSize));
            using var rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

            // Create input tensor [1, 3, 640, 640] normalized [0, 1]
            var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });
            var matIndexer = rgb.GetGenericIndexer<Vec3b>();

            for (int y = 0; y < ModelInputSize; y++)
            {
                for (int x = 0; x < ModelInputSize; x++)
                {
                    var pixel = matIndexer[y, x];
                    tensor[0, 0, y, x] = pixel.Item0 / 255.0f;
                    tensor[0, 1, y, x] = pixel.Item1 / 255.0f;
                    tensor[0, 2, y, x] = pixel.Item2 / 255.0f;
                }
            }

            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("images", tensor) };
            using var outputs = _wheelSession.Run(inputs);
            var outputTensor = outputs.First().AsTensor<float>(); // Shape: [1, 6, 8400]

            float scaleX = (float)origW / ModelInputSize;
            float scaleY = (float)origH / ModelInputSize;

            int numAnchors = outputTensor.Dimensions[2];
            var candidates = new List<Detection>();

            for (int a = 0; a < numAnchors; a++)
            {
                float scoreCab = outputTensor[0, 4, a];
                float scoreWheel = outputTensor[0, 5, a];

                float maxScore = Math.Max(scoreCab, scoreWheel);
                if (maxScore < 0.25f) continue;

                int clsId = scoreCab > scoreWheel ? 0 : 1;
                float cx = outputTensor[0, 0, a];
                float cy = outputTensor[0, 1, a];
                float bw = outputTensor[0, 2, a];
                float bh = outputTensor[0, 3, a];

                float x1 = (cx - bw / 2f) * scaleX;
                float y1 = (cy - bh / 2f) * scaleY;
                float w = bw * scaleX;
                float h = bh * scaleY;

                candidates.Add(new Detection
                {
                    ClassId = clsId,
                    Confidence = maxScore,
                    Bbox = new Rect2f(x1, y1, w, h)
                });
            }

            // Simple Non-Maximum Suppression (NMS)
            return NonMaximumSuppression(candidates, 0.45f);
        }

        private static List<Detection> NonMaximumSuppression(List<Detection> boxes, float iouThresh)
        {
            var result = new List<Detection>();
            var sorted = boxes.OrderByDescending(b => b.Confidence).ToList();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                result.Add(best);
                sorted.RemoveAt(0);

                sorted.RemoveAll(b => b.ClassId == best.ClassId && ComputeIoU(b.Bbox, best.Bbox) > iouThresh);
            }
            return result;
        }

        private static float ComputeIoU(Rect2f a, Rect2f b)
        {
            float x1 = Math.Max(a.Left, b.Left);
            float y1 = Math.Max(a.Top, b.Top);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);

            float interArea = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            float unionArea = (a.Width * a.Height) + (b.Width * b.Height) - interArea;
            return unionArea <= 0 ? 0 : interArea / unionArea;
        }

        /// <summary>
        /// Estimates the ground contacts with Dynamic Track Width Scaling and Bumper-Level Y-Anchor.
        /// </summary>
        public GroundContact EstimateVehicleGroundContact(
            Rect2f vBbox,
            string vClass,
            List<Detection> steerTires,
            Rect2f? cabFaceBbox,
            RoadCalibration calib)
        {
            double baseTrackWm = vClass.ToLower() switch
            {
                "truck" or "bus" or "train" => 2.05,
                "car" or "van" or "suv" => 1.65,
                _ => 0.90
            };

            double trackWm = baseTrackWm;

            // Physical Track Width Constraint based on BEV Cab Width
            if (cabFaceBbox.HasValue)
            {
                var cab = cabFaceBbox.Value;
                var bevCabL = calib.ProjectToBev(new Point2D(cab.Left, cab.Bottom));
                var bevCabR = calib.ProjectToBev(new Point2D(cab.Right, cab.Bottom));
                double cabWm = Math.Abs(bevCabR.X - bevCabL.X) / calib.PixelsPerMeterX;

                if (cabWm < 1.95)
                {
                    // Pickup truck / compact truck / van (even if classified by YOLO as "truck")
                    trackWm = Math.Min(baseTrackWm, Math.Max(1.40, cabWm * 0.88));
                }
                else
                {
                    // Heavy commercial truck / bus
                    trackWm = Math.Min(baseTrackWm, cabWm * 0.90);
                }
            }

            double trackWbev = trackWm * calib.PixelsPerMeterX;
            Point2D? pLeft = null;
            Point2D? pRight = null;
            Point2D? bevLeft = null;
            Point2D? bevRight = null;

            if (steerTires.Count >= 2)
            {
                var sorted = steerTires.OrderBy(t => t.Bbox.X + t.Bbox.Width / 2f).ToList();
                var tL = sorted.First();
                var tR = sorted.Last();
                double avgY = (tL.Bbox.Bottom + tR.Bbox.Bottom) / 2.0;
                pLeft = new Point2D(tL.Bbox.X + tL.Bbox.Width / 2.0, avgY);
                pRight = new Point2D(tR.Bbox.X + tR.Bbox.Width / 2.0, avgY);
                bevLeft = calib.ProjectToBev(pLeft.Value);
                bevRight = calib.ProjectToBev(pRight.Value);
            }
            else if (steerTires.Count == 1)
            {
                var t = steerTires[0];
                double tx = t.Bbox.X + t.Bbox.Width / 2.0;
                double ty = t.Bbox.Bottom;

                bool anchorIsLeft = cabFaceBbox.HasValue
                    ? tx < (cabFaceBbox.Value.Left + cabFaceBbox.Value.Right) / 2.0
                    : (tx - vBbox.Left) <= (vBbox.Right - tx);

                if (anchorIsLeft)
                {
                    pLeft = new Point2D(tx, ty);
                    bevLeft = calib.ProjectToBev(pLeft.Value);
                    bevRight = new Point2D(bevLeft.Value.X + trackWbev, bevLeft.Value.Y);
                    var proj = calib.ProjectFromBev(bevRight.Value);

                    // Enforce physical constraints: anchor to bumper ground level & clamp inside cab bounds
                    double refY = cabFaceBbox.HasValue ? cabFaceBbox.Value.Bottom : ty;
                    double projY = Math.Clamp(proj.Y, refY - 8.0, refY + 8.0);
                    double maxX = cabFaceBbox.HasValue ? cabFaceBbox.Value.Right + 10.0 : vBbox.Right + 10.0;
                    pRight = new Point2D(Math.Min(proj.X, maxX), projY);
                }
                else
                {
                    pRight = new Point2D(tx, ty);
                    bevRight = calib.ProjectToBev(pRight.Value);
                    bevLeft = new Point2D(bevRight.Value.X - trackWbev, bevRight.Value.Y);
                    var proj = calib.ProjectFromBev(bevLeft.Value);

                    double refY = cabFaceBbox.HasValue ? cabFaceBbox.Value.Bottom : ty;
                    double projY = Math.Clamp(proj.Y, refY - 8.0, refY + 8.0);
                    double minX = cabFaceBbox.HasValue ? cabFaceBbox.Value.Left - 10.0 : vBbox.Left - 10.0;
                    pLeft = new Point2D(Math.Max(proj.X, minX), projY);
                }
            }

            // Robust Fallback if no tires were detected
            if (!pLeft.HasValue || !pRight.HasValue)
            {
                if (cabFaceBbox.HasValue)
                {
                    double cabCx = (cabFaceBbox.Value.Left + cabFaceBbox.Value.Right) / 2.0;
                    double cabBottomY = cabFaceBbox.Value.Bottom;
                    var bevCabCenter = calib.ProjectToBev(new Point2D(cabCx, cabBottomY));
                    double halfW = trackWbev / 2.0;
                    bevLeft = new Point2D(bevCabCenter.X - halfW, bevCabCenter.Y);
                    bevRight = new Point2D(bevCabCenter.X + halfW, bevCabCenter.Y);
                    pLeft = calib.ProjectFromBev(bevLeft.Value);
                    pRight = calib.ProjectFromBev(bevRight.Value);
                }
                else
                {
                    double vCx = (vBbox.Left + vBbox.Right) / 2.0;
                    var bevCenter = calib.ProjectToBev(new Point2D(vCx, vBbox.Bottom));
                    double halfW = trackWbev / 2.0;
                    bevLeft = new Point2D(bevCenter.X - halfW, bevCenter.Y);
                    bevRight = new Point2D(bevCenter.X + halfW, bevCenter.Y);
                    pLeft = calib.ProjectFromBev(bevLeft.Value);
                    pRight = calib.ProjectFromBev(bevRight.Value);
                }
            }

            var pCenter = new Point2D((pLeft.Value.X + pRight.Value.X) / 2.0, (pLeft.Value.Y + pRight.Value.Y) / 2.0);
            var bevCenterPt = new Point2D((bevLeft!.Value.X + bevRight!.Value.X) / 2.0, (bevLeft.Value.Y + bevRight.Value.Y) / 2.0);

            return new GroundContact
            {
                LeftWheel = pLeft.Value,
                RightWheel = pRight.Value,
                CenterBottom = pCenter,
                BevLeftWheel = bevLeft.Value,
                BevRightWheel = bevRight.Value,
                BevCenterBottom = bevCenterPt,
                CabFaceBbox = cabFaceBbox
            };
        }

        /// <summary>
        /// Evaluates whether the vehicle is straddling either lane line accounting for finite tire patch width.
        /// </summary>
        public StraddleAssessment AssessLaneStraddling(
            GroundContact contact,
            RoadCalibration calib,
            string vClass = "truck",
            double toleranceCm = 0.0)
        {
            if (calib.LeftLane.Count < 2 || calib.RightLane.Count < 2)
            {
                return new StraddleAssessment
                {
                    IsStraddling = false,
                    StatusLabel = "NO_CALIB",
                    Details = "Lane boundaries not configured"
                };
            }

            double tireHalfWidthCm = (vClass.ToLower() == "truck" || vClass.ToLower() == "bus")
                ? TruckTireHalfWidthCm
                : CarTireHalfWidthCm;

            var leftLaneBev = calib.LeftLane.Select(calib.ProjectToBev).ToList();
            var rightLaneBev = calib.RightLane.Select(calib.ProjectToBev).ToList();

            double leftWheelY = contact.BevLeftWheel.Y;
            double rightWheelY = contact.BevRightWheel.Y;

            double leftLineX = InterpolateXOnLine(leftLaneBev[0], leftLaneBev[1], leftWheelY) ?? 0.0;
            double rightLineX = InterpolateXOnLine(rightLaneBev[0], rightLaneBev[1], rightWheelY) ?? (calib.BevWidth - 1);

            double pxPerCm = calib.PixelsPerMeterX / 100.0;

            // Distance from tire center to lane line (positive = inside lane)
            double leftCenterDistCm = (contact.BevLeftWheel.X - leftLineX) / pxPerCm;
            double rightCenterDistCm = (rightLineX - contact.BevRightWheel.X) / pxPerCm;

            // Clearance from outer tire contact patch to lane line (negative = tire edge penetrates lane line)
            double leftClearanceCm = leftCenterDistCm - tireHalfWidthCm;
            double rightClearanceCm = rightCenterDistCm - tireHalfWidthCm;

            // Check if vehicle is approaching evaluation corridor
            double avgBevY = (leftWheelY + rightWheelY) / 2.0;
            if (avgBevY < -150.0)
            {
                double distM = Math.Abs(avgBevY) / calib.PixelsPerMeterY;
                return new StraddleAssessment
                {
                    IsStraddling = false,
                    StatusLabel = "APPROACHING",
                    ViolationSide = null,
                    LeftWheelClearanceCm = leftClearanceCm,
                    RightWheelClearanceCm = rightClearanceCm,
                    StraddleDepthCm = 0.0,
                    Details = $"Vehicle approaching corridor ({distM:F1}m away)"
                };
            }

            bool straddleLeft = leftClearanceCm < -toleranceCm;
            bool straddleRight = rightClearanceCm < -toleranceCm;
            double depthCm = 0.0;

            if (straddleLeft) depthCm = Math.Max(depthCm, Math.Abs(leftClearanceCm));
            if (straddleRight) depthCm = Math.Max(depthCm, Math.Abs(rightClearanceCm));
            depthCm = Math.Min(500.0, depthCm);

            if (straddleLeft && straddleRight)
            {
                return new StraddleAssessment
                {
                    IsStraddling = true,
                    StatusLabel = "OUT_OF_LANE",
                    ViolationSide = "both",
                    LeftWheelClearanceCm = leftClearanceCm,
                    RightWheelClearanceCm = rightClearanceCm,
                    StraddleDepthCm = depthCm,
                    Details = $"Crossed both lanes (L: {leftClearanceCm:F1}cm, R: {rightClearanceCm:F1}cm)"
                };
            }
            if (straddleLeft)
            {
                return new StraddleAssessment
                {
                    IsStraddling = true,
                    StatusLabel = "STRADDLE_LEFT",
                    ViolationSide = "left",
                    LeftWheelClearanceCm = leftClearanceCm,
                    RightWheelClearanceCm = rightClearanceCm,
                    StraddleDepthCm = depthCm,
                    Details = $"Left tire crossed lane by {Math.Abs(leftClearanceCm):F1}cm"
                };
            }
            if (straddleRight)
            {
                return new StraddleAssessment
                {
                    IsStraddling = true,
                    StatusLabel = "STRADDLE_RIGHT",
                    ViolationSide = "right",
                    LeftWheelClearanceCm = leftClearanceCm,
                    RightWheelClearanceCm = rightClearanceCm,
                    StraddleDepthCm = depthCm,
                    Details = $"Right tire crossed lane by {Math.Abs(rightClearanceCm):F1}cm"
                };
            }

            return new StraddleAssessment
            {
                IsStraddling = false,
                StatusLabel = "IN_LANE",
                LeftWheelClearanceCm = leftClearanceCm,
                RightWheelClearanceCm = rightClearanceCm,
                StraddleDepthCm = 0.0,
                Details = $"Inside lane (Clearance L: {leftClearanceCm:F1}cm, R: {rightClearanceCm:F1}cm)"
            };
        }

        private static double? InterpolateXOnLine(Point2D p1, Point2D p2, double y)
        {
            if (Math.Abs(p2.Y - p1.Y) < 1e-6) return null;
            double t = (y - p1.Y) / (p2.Y - p1.Y);
            return p1.X + t * (p2.X - p1.X);
        }

        public void Dispose()
        {
            _wheelSession?.Dispose();
        }
    }
}
