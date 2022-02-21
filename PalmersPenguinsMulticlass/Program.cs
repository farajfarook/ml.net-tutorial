// See https://aka.ms/new-console-template for more information

using PalmerPenguins;
using PalmersPenguinsMulticlass.Models;
using PalmersPenguinsMulticlass.Trainers;

var dataFile = Path.Combine(AppContext.BaseDirectory, "Data", "penguins.csv");

var newSample = new PalmerPenguinsData
{
    Island = "Torgersen",
    CulmenDepth = 18.7f,
    CulmenLength = 39.3f,
    FliperLength = 180,
    BodyMass = 3700,
    Sex = "MALE"
};


var trainers = new List<ITrainerBase>
{
    new LbfgsMaximumEntropyTrainer(),
    new NaiveBayesTrainer(),
    new OneVersusAllTrainer(),
    new SdcaMaximumEntropyTrainer(),
    new SdcaNonCalibratedTrainer()
};

trainers.ForEach(trainer =>
{
    Console.WriteLine("*******************************");
    Console.WriteLine($"{ trainer.Name }");
    Console.WriteLine("*******************************");

    trainer.Fit(dataFile);

    var modelMetrics = trainer.Evaluate();

    Console.WriteLine($"Macro Accuracy: {modelMetrics.MacroAccuracy:#.##}{Environment.NewLine}" +
                      $"Micro Accuracy: {modelMetrics.MicroAccuracy:#.##}{Environment.NewLine}" +
                      $"Log Loss: {modelMetrics.LogLoss:#.##}{Environment.NewLine}" +
                      $"Log Loss Reduction: {modelMetrics.LogLossReduction:#.##}{Environment.NewLine}");

    trainer.Save();

    var predictor = new Predictor(trainer.Name);
    var prediction = predictor.Predict(newSample);
    Console.WriteLine("------------------------------");
    Console.WriteLine($"Prediction: {prediction.PredictedLabel:#.##}");
    Console.WriteLine("------------------------------");
});