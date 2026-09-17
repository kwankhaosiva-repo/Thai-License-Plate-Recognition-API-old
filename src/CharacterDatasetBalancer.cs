using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;
using OpenCvSharp;

namespace ThaiLPR.DataPreparation
{
    /// <summary>
    /// C# port of src/balance_and_augment_characters.py
    /// 
    /// Merges character crops from:
    ///   1. datasets/Thai/thai_character_crops/by_character_square/ (initial verified crops)
    ///   2. datasets/Thai/thai_character_crops/candidates/ (cleaned user-curated candidates)
    /// 
    /// Performs targeted offline data augmentation for rare classes (< 60 samples)
    /// using realistic affine rotations, translation, brightness/contrast jitter,
    /// and subtle Gaussian blur or sharpening.
    /// 
    /// Populates:
    ///   datasets/Thai/thai_character_crops/splits/train/<class>/
    ///   datasets/Thai/thai_character_crops/splits/valid/<class>/
    /// 
    /// Saves:
    ///   weights/char_classifier_map.json (mapping integer ID 0..49 to character name)
    /// </summary>
    public class CharacterDatasetBalancer
    {
        private readonly string _baseDir;
        private readonly string _candidatesDir;
        private readonly string _squareDir;
        private readonly string _splitsDir;
        private readonly string _weightsDir;
        private readonly string _mapPath;

        private const int TargetMinSamples = 60;
        private const double TrainRatio = 0.85;

        private static readonly Random Rng = new Random(42);

        public CharacterDatasetBalancer(string projectRoot)
        {
            // Support both datasets/Thai/thai_character_crops and datasets/thai_character_crops
            string thaiCropPath = Path.Combine(projectRoot, "datasets", "Thai", "thai_character_crops");
            if (!Directory.Exists(thaiCropPath))
            {
                thaiCropPath = Path.Combine(projectRoot, "datasets", "thai_character_crops");
            }

            _baseDir = thaiCropPath;
            _candidatesDir = Path.Combine(_baseDir, "candidates");
            _squareDir = Path.Combine(_baseDir, "by_character_square");
            _splitsDir = Path.Combine(_baseDir, "splits");
            _weightsDir = Path.Combine(projectRoot, "weights");
            _mapPath = Path.Combine(_weightsDir, "char_classifier_map.json");
        }

        /// <summary>
        /// Applies realistic photometric and geometric augmentations to a character crop (OpenCV Mat).
        /// </summary>
        public static Mat AugmentImage(Mat src)
        {
            int h = src.Height;
            int w = src.Width;

            // Estimate background color from 4 corners
            Vec3b c1 = src.At<Vec3b>(0, 0);
            Vec3b c2 = src.At<Vec3b>(0, w - 1);
            Vec3b c3 = src.At<Vec3b>(h - 1, 0);
            Vec3b c4 = src.At<Vec3b>(h - 1, w - 1);

            Scalar avgBg = new Scalar(
                (c1.Item0 + c2.Item0 + c3.Item0 + c4.Item0) / 4.0,
                (c1.Item1 + c2.Item1 + c3.Item1 + c4.Item1) / 4.0,
                (c1.Item2 + c2.Item2 + c3.Item2 + c4.Item2) / 4.0
            );

            // 1. Random Rotation (-7 to +7 degrees) and Translation (-3 to +3 pixels)
            double angle = (Rng.NextDouble() * 14.0) - 7.0;
            double tx = (Rng.NextDouble() * 6.0) - 3.0;
            double ty = (Rng.NextDouble() * 6.0) - 3.0;

            Point2f center = new Point2f(w / 2.0f, h / 2.0f);
            using var rotMat = Cv2.GetRotationMatrix2D(center, angle, 1.0);
            rotMat.Set(0, 2, rotMat.At<double>(0, 2) + tx);
            rotMat.Set(1, 2, rotMat.At<double>(1, 2) + ty);

            var transformed = new Mat();
            Cv2.WarpAffine(src, transformed, rotMat, new Size(w, h), InterpolationFlags.Cubic, BorderTypes.Constant, avgBg);

            // 2. Brightness & Contrast jitter (0.75 - 1.25)
            // out = contrast * in + (brightness - 128)
            double contrastFactor = 0.75 + (Rng.NextDouble() * 0.50);   // 0.75 - 1.25
            double brightnessDelta = ((Rng.NextDouble() * 0.50) - 0.25) * 50.0; // -12.5 to +12.5 px shift

            var adjusted = new Mat();
            transformed.ConvertTo(adjusted, MatType.CV_8UC3, contrastFactor, brightnessDelta);
            transformed.Dispose();

            // 3. Occasional slight Gaussian blur or unsharp mask sharpening
            double roll = Rng.NextDouble();
            if (roll < 0.35)
            {
                // Subtle Gaussian blur
                var blurred = new Mat();
                Cv2.GaussianBlur(adjusted, blurred, new Size(3, 3), 0.6);
                adjusted.Dispose();
                return blurred;
            }
            else if (roll < 0.70)
            {
                // Subtle sharpness filter
                var blurred = new Mat();
                Cv2.GaussianBlur(adjusted, blurred, new Size(0, 0), 1.0);
                var sharpened = new Mat();
                Cv2.AddWeighted(adjusted, 1.5, blurred, -0.5, 0, sharpened);
                blurred.Dispose();
                adjusted.Dispose();
                return sharpened;
            }

            return adjusted;
        }

        /// <summary>
        /// Balances and splits character crops into train/valid sets.
        /// </summary>
        public void BalanceAndAugment()
        {
            Console.WriteLine("==================================================================");
            Console.WriteLine("THAI CHARACTER DATASET BALANCING & TARGETED AUGMENTATION (C#)");
            Console.WriteLine("==================================================================");

            var candClasses = Directory.Exists(_candidatesDir)
                ? Directory.GetDirectories(_candidatesDir).Select(Path.GetFileName)
                : Enumerable.Empty<string>();

            var squareClasses = Directory.Exists(_squareDir)
                ? Directory.GetDirectories(_squareDir).Select(Path.GetFileName)
                : Enumerable.Empty<string>();

            var allUniqueClasses = candClasses.Union(squareClasses).Where(s => !string.IsNullOrEmpty(s)).ToList();

            Console.WriteLine($"Discovered {allUniqueClasses.Count} unique character classes.");

            // Digits first (0..9), then Thai consonants (ก..ฮ)
            var digits = allUniqueClasses.Where(c => c.All(char.IsDigit)).OrderBy(c => c).ToList();
            var consonants = allUniqueClasses.Where(c => !c.All(char.IsDigit)).OrderBy(c => c).ToList();
            var orderedClasses = digits.Concat(consonants).ToList();

            // Save JSON mapping (ID -> Class Name)
            var charMap = new Dictionary<string, string>();
            for (int i = 0; i < orderedClasses.Count; i++)
            {
                charMap[i.ToString()] = orderedClasses[i];
            }

            Directory.CreateDirectory(_weightsDir);
            var jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
            };
            File.WriteAllText(_mapPath, JsonSerializer.Serialize(charMap, jsonOptions));
            Console.WriteLine($"Saved {charMap.Count} classes mapping to: {_mapPath}");

            // Reset splits/train and splits/valid
            string trainDir = Path.Combine(_splitsDir, "train");
            string validDir = Path.Combine(_splitsDir, "valid");

            if (Directory.Exists(trainDir)) Directory.Delete(trainDir, true);
            if (Directory.Exists(validDir)) Directory.Delete(validDir, true);

            Directory.CreateDirectory(trainDir);
            Directory.CreateDirectory(validDir);

            int totalTrain = 0;
            int totalValid = 0;
            int totalAugmented = 0;

            Console.WriteLine("\nProcessing each class (merging + augmenting rare classes)...");

            foreach (var ch in orderedClasses)
            {
                var realImages = new List<string>();

                string candChDir = Path.Combine(_candidatesDir, ch);
                if (Directory.Exists(candChDir))
                {
                    realImages.AddRange(Directory.GetFiles(candChDir, "*.jpg"));
                    realImages.AddRange(Directory.GetFiles(candChDir, "*.png"));
                }

                string sqChDir = Path.Combine(_squareDir, ch);
                if (Directory.Exists(sqChDir))
                {
                    var sqFiles = Directory.GetFiles(sqChDir, "*.jpg").Concat(Directory.GetFiles(sqChDir, "*.png"));
                    var existingNames = new HashSet<string>(realImages.Select(Path.GetFileName));
                    foreach (var f in sqFiles)
                    {
                        if (!existingNames.Contains(Path.GetFileName(f)))
                        {
                            realImages.Add(f);
                        }
                    }
                }

                int nReal = realImages.Count;
                string chTrainDir = Path.Combine(trainDir, ch);
                string chValidDir = Path.Combine(validDir, ch);
                Directory.CreateDirectory(chTrainDir);
                Directory.CreateDirectory(chValidDir);

                // Shuffle real images
                Shuffle(realImages);

                int nVal = nReal >= 2 ? Math.Max(1, (int)Math.Round(nReal * (1.0 - TrainRatio))) : 0;
                int nTrain = nReal - nVal;

                var realTrain = realImages.Take(nTrain).ToList();
                var realVal = realImages.Skip(nTrain).ToList();

                // Copy real train images
                foreach (var srcPath in realTrain)
                {
                    string dest = Path.Combine(chTrainDir, Path.GetFileName(srcPath));
                    File.Copy(srcPath, dest, true);
                    totalTrain++;
                }

                // Copy real val images
                foreach (var srcPath in realVal)
                {
                    string dest = Path.Combine(chValidDir, Path.GetFileName(srcPath));
                    File.Copy(srcPath, dest, true);
                    totalValid++;
                }

                // Augment rare classes to hit TargetMinSamples
                int currentTrainCount = realTrain.Count;
                int augmentedAdded = 0;

                if (currentTrainCount < TargetMinSamples && realImages.Count > 0)
                {
                    int deficit = TargetMinSamples - currentTrainCount;
                    for (int i = 0; i < deficit; i++)
                    {
                        string srcImgPath = realImages[i % realImages.Count];
                        using var srcMat = Cv2.ImRead(srcImgPath, ImreadModes.Color);
                        if (srcMat.Empty()) continue;

                        using var augMat = AugmentImage(srcMat);
                        string augFileName = $"aug_{Path.GetFileNameWithoutExtension(srcImgPath)}_v{i}.jpg";
                        string destPath = Path.Combine(chTrainDir, augFileName);

                        Cv2.ImWrite(destPath, augMat, new[] { (int)ImwriteFlags.JpegQuality, 95 });

                        augmentedAdded++;
                        totalTrain++;
                        totalAugmented++;
                    }
                }
            }

            Console.WriteLine("\n==================================================================");
            Console.WriteLine("BALANCING & AUGMENTATION SUMMARY (C#)");
            Console.WriteLine("==================================================================");
            Console.WriteLine($"Total Character Classes          : {orderedClasses.Count}");
            Console.WriteLine($"Total Training Samples           : {totalTrain}");
            Console.WriteLine($"Total Validation Samples         : {totalValid}");
            Console.WriteLine($"Total Augmented Samples Created  : {totalAugmented}");
            Console.WriteLine("==================================================================\n");
        }

        private static void Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = Rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
