using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    class Program
    {
        private static MLContext _ctx = new MLContext();
        private static string _dataFile = Path.Combine("Data", "yelp_labelled.txt");
        static void Main(string[] args)
        {
            var splitData = _ctx.Data.TrainTestSplit(_ctx.Data.LoadFromTextFile<SentimentData>(_dataFile));
            var pipe = _ctx.Transforms.Text.FeaturizeText("FText", "Text")
                .Append(_ctx.Transforms.Concatenate("Features", "FText"))
                .Append(_ctx.BinaryClassification.Trainers.SdcaLogisticRegression());

            var model = pipe.Fit(splitData.TrainSet);
            var metrics = _ctx.BinaryClassification.Evaluate(model.Transform(splitData.TestSet));
            
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");
            
            var sentiments = new[]
            {
                new SentimentData
                {
                    Text = "This was a horrible meal"
                },
                new SentimentData
                {
                    Text = "I love this spaghetti."
                }
            };

            var resp = model.Transform(_ctx.Data.LoadFromEnumerable(sentiments));
            var predictedResults = _ctx.Data.CreateEnumerable<SentimentPrediction>(resp, false);                 
            
            foreach (var prediction  in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.Text} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
            }
        }
    }

    class SentimentData
    {
        [LoadColumn(0)]
        public string Text { get; set; }
        [LoadColumn(1), ColumnName("Label")]
        public bool Sentiment { get; set; }
    }

    class SentimentPrediction: SentimentData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }
}