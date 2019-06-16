using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string _trainPath = Path.Combine("Data", "issues_train.tsv");
        private static string _testPath = Path.Combine("Data", "issues_test.tsv");
        private static string _modelPath = "model.zip";

        private static MLContext _ctx = new MLContext();
        private static ITransformer model;

        static void Main(string[] args)
        {
            if (!File.Exists(_modelPath))
            {
                var trainData = _ctx.Data.LoadFromTextFile<Issue>(_trainPath, hasHeader: true);

                var pipe = _ctx.Transforms.Conversion.MapValueToKey("Label", "Area")
                    .Append(_ctx.Transforms.Text.FeaturizeText("FTitle", "Title"))
                    .Append(_ctx.Transforms.Text.FeaturizeText("FDescription", "Description"))
                    .Append(_ctx.Transforms.Concatenate("Features", "FTitle", "FDescription"))
                    .AppendCacheCheckpoint(_ctx);

                var trainPipe = pipe.Append(_ctx.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                    .Append(_ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                Console.Write("Training...");
                model = trainPipe.Fit(trainData);
                Console.WriteLine("Done.");
                _ctx.Model.Save(model, trainData.Schema, _modelPath);
            }
            else
            {
                model = _ctx.Model.Load(_modelPath, out var scheme);
            }

            var testData = _ctx.Data.LoadFromTextFile<Issue>(_testPath, hasHeader: true);
            var testMetrics = _ctx.MulticlassClassification.Evaluate(model.Transform(testData));
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
        }
    }

    class Issue
    {
        [LoadColumn(0)]
        public string Id { get; set; }
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }
    }

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}