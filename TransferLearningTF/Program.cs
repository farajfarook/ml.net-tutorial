using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TransferLearningTF
{
    class Program
    {
        private static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        private static readonly string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags.tsv");
        private static readonly string _predictImageListTsv = Path.Combine(_assetsPath, "inputs-predict", "data", "image_list.tsv");
        private static readonly string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "data");
        private static readonly string _predictImagesFolder = Path.Combine(_assetsPath, "inputs-predict", "data");
        private static readonly string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "toaster3.jpg");
        private static readonly string _inceptionPb = Path.Combine(_assetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
        private static readonly string _inputImageClassifierZip = Path.Combine(_assetsPath, "inputs-predict", "imageClassifier.zip");
        private static readonly string _outputImageClassifierZip = Path.Combine(_assetsPath, "outputs", "imageClassifier.zip");
        private static string LabelTokey = nameof(LabelTokey);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        private static MLContext _ctx = new MLContext(1);
        
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }
        
        static void Main(string[] args)
        {
            var model = ReuseAndTuneInceptionModel(_trainTagsTsv, _trainImagesFolder, _inceptionPb, _outputImageClassifierZip);
            
            //----
            var imageData = ReadFromTsv(_predictImageListTsv, _predictImagesFolder);
            var imageDataView = _ctx.Data.LoadFromEnumerable(imageData);
            
            var predictions = model.Transform(imageDataView);
            var imagePredictionData = _ctx.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);
            
            DisplayResults(imagePredictionData);
            
            //----
            var singleImageData = new ImageData
            {
                ImagePath = _predictSingleImage
            };
            
            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = _ctx.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(singleImageData);
            
            Console.WriteLine($"Image: {Path.GetFileName(singleImageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
        }


        public static ITransformer ReuseAndTuneInceptionModel(string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            var pipe = _ctx.Transforms.Conversion.MapValueToKey(LabelTokey, "Label")
                .Append(_ctx.Transforms.LoadImages("input", imagesFolder, nameof(ImageData.ImagePath)))
                .Append(_ctx.Transforms.ResizeImages("input", InceptionSettings.ImageWidth,
                    InceptionSettings.ImageHeight, "input"))
                .Append(_ctx.Transforms.ExtractPixels("input", "input",
                    interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean,
                    scaleImage: InceptionSettings.Scale))
                .Append(_ctx.Model.LoadTensorFlowModel(inputModelLocation)
                    .ScoreTensorFlowModel("softmax2_pre_activation", "input", true))
                .Append(_ctx.MulticlassClassification.Trainers.LbfgsMaximumEntropy(LabelTokey, "softmax2_pre_activation"))
                .Append(_ctx.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
                .AppendCacheCheckpoint(_ctx);

            var data = _ctx.Data.LoadFromTextFile<ImageData>(dataLocation);
            return pipe.Fit(data);            
        }
        
        
        
        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }
        
        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
                .Select(line => line.Split('\t'))
                .Select(line => new ImageData()
                {
                    ImagePath = Path.Combine(folder, line[0])
                });
        }
    }
    
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }
    
    public class ImagePrediction : ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }
    
}