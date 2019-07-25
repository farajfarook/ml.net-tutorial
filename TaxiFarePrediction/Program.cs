using System;
using System.IO;
using Microsoft.ML;

namespace TaxiFarePrediction
{
    class Program
    {
        private static MLContext _ctx = new MLContext(1);
        private static string _trainFile = Path.Combine("Data", "taxi-fare-train.csv");
        private static string _testFile = Path.Combine("Data", "taxi-fare-test.csv");        

        static void Main(string[] args)
        {
            var prePipe = _ctx.Transforms.Categorical.OneHotEncoding("VendorIdEnc", "VendorId")
                .Append(_ctx.Transforms.Categorical.OneHotEncoding("PaymentTypeEnc", "PaymentType"))
                .Append(_ctx.Transforms.Categorical.OneHotEncoding("RateCodeEnc", "RateCode"))
                .Append(_ctx.Transforms.Concatenate("Features",
                    "VendorIdEnc", "RateCodeEnc", "Passengers", "TripTime", "Distance", "PaymentTypeEnc"));
                
            Console.WriteLine("FastTree");
            var pipe = prePipe.Append(_ctx.Regression.Trainers.FastTree());
            var trainData = _ctx.Data.LoadFromTextFile<TaxiTrip>(_trainFile, ',', true);
            var model = pipe.Fit(trainData);
            Evaluates(model);
            
            
            Console.WriteLine("LightGbm");
            var pipe2 = prePipe.Append(_ctx.Regression.Trainers.LightGbm());
            var trainData2 = _ctx.Data.LoadFromTextFile<TaxiTrip>(_trainFile, ',', true);
            var model2 = pipe2.Fit(trainData2);
            Evaluates(model2);
        }

        private static void Evaluates(ITransformer model)
        {
            var testData = _ctx.Data.LoadFromTextFile<TaxiTrip>(_testFile, ',', true);
            
            var metrics =_ctx.Regression.Evaluate(model.Transform(testData));
            
            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS error: {metrics.RootMeanSquaredError:0.##}");
            
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                Passengers = 1,
                TripTime = 1140,
                Distance = 3.75f,
                PaymentType = "CRD",
                Fare = 0 // To predict. Actual/Observed = 15.5
            };

            var eng = _ctx.Model.CreatePredictionEngine<TaxiTrip, TaxiTripPrediction>(model);

            var pred = eng.Predict(taxiTripSample);
            
            Console.WriteLine(pred.Fare);
        }
    }
}