using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MovieRecommender
{
    class Program
    {
        private static MLContext _ctx = new MLContext(0);
        static void Main(string[] args)
        {
            
            var pipe =
                _ctx.Transforms.Conversion.MapValueToKey("userIdEnc", "userId")
                    .Append(_ctx.Transforms.Conversion.MapValueToKey("movieIdEnc", "movieId"))
                    .Append(_ctx.Recommendation().Trainers.MatrixFactorization(new MatrixFactorizationTrainer.Options()
                    {
                        MatrixColumnIndexColumnName = "userIdEnc",
                        MatrixRowIndexColumnName = "movieIdEnc", 
                        LabelColumnName = "Label",
                        NumberOfIterations = 20,
                        ApproximationRank = 100
                    }));
            
            
            var data = LoadData(_ctx);

            var model = pipe.Fit(data.training);
            
            var prediction = model.Transform(data.test);

            var metrics = _ctx.Regression.Evaluate(prediction);

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError);
            Console.WriteLine("RSquared: " + metrics.RSquared);

        }
        
        public static (IDataView training, IDataView test) LoadData(MLContext mlContext)
        {
            var trainingDataPath = Path.Combine("Data", "recommendation-ratings-train.csv");
            var testDataPath = Path.Combine("Data", "recommendation-ratings-test.csv");
            
            var trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            var testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');
            return (trainingDataView, testDataView);
        }
    }
    
    public class MovieRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float movieId;
        [LoadColumn(2)]
        public float Label;
    }
    
    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }
}