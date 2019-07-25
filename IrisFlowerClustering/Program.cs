using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IrisFlowerClustering
{
    class Program
    {
        private static MLContext _ctx = new MLContext(0);
        private static string _dataFile = Path.Combine("Data", "iris.data");
        private static string _modelFile = "model.zip";

        internal static readonly IrisData Setosa = new IrisData
        {
            SepalLength = 5.1f,
            SepalWidth = 3.5f,
            PetalLength = 1.4f,
            PetalWidth = 0.2f
        };

        static void Main(string[] args)
        {
            var pipe = _ctx
                .Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(_ctx.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

            var dataView = _ctx.Data.LoadFromTextFile<IrisData>(_dataFile, ',');
            var model = pipe.Fit(dataView);

            using (var fileStream = new FileStream(_modelFile, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                _ctx.Model.Save(model, dataView.Schema, fileStream);
            }

            var predictor = _ctx.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

            var prediction = predictor.Predict(Setosa);

            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
    }

    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}