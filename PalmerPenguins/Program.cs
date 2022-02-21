// See https://aka.ms/new-console-template for more information

using PalmerPenguins;
using PalmerPenguins.Models;
using PalmerPenguins.Trainers;

var binaryFile = Path.Combine(AppContext.BaseDirectory, "Data", "penguins_binary.csv");

var newSample = new PalmerPenguinsBinaryData
{
    BIllDepth = 1.2f,
    BillLength = 1.1f
};

var trainers = new List<ITrainerBase>
{
    new LbfgsLogisticRegressionTrainer(),
    new AveragedPerceptronTrainer(),
    new PriorTrainer(),
    new SdcaLogisticRegressionTrainer(),
    new SdcaNonCalibratedTrainer(),
    new SgdCalibratedTrainer(),
    new SgdNonCalibratedTrainer()
};

trainers.ForEach(trainer =>
{
    Console.WriteLine("*******************************");
    Console.WriteLine($"{ trainer.Name }");
    Console.WriteLine("*******************************");

    trainer.Fit(binaryFile);

    var modelMetrics = trainer.Evaluate();
    Console.WriteLine($"Accuracy: {modelMetrics.Accuracy:0.##}{Environment.NewLine}" +
                      $"F1 Score: {modelMetrics.F1Score:#.##}{Environment.NewLine}" +
                      $"Positive Precision: {modelMetrics.PositivePrecision:#.##}{Environment.NewLine}" +
                      $"Negative Precision: {modelMetrics.NegativePrecision:0.##}{Environment.NewLine}" +
                      $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                      $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}" +
                      $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:#.##}{Environment.NewLine}");

    trainer.Save();

    var predictor = new Predictor(trainer.Name);
    var prediction = predictor.Predict(newSample);
    Console.WriteLine("------------------------------");
    Console.WriteLine($"Prediction: {prediction.PredictedLabel:#.##}");
    Console.WriteLine("------------------------------");
    
});