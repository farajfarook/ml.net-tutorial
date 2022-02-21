// See https://aka.ms/new-console-template for more information

using Placement;
using Placement.Models;
using Placement.Trainers;

var dataFile = Path.Combine(AppContext.BaseDirectory, "Data", "Placement_Data_Full_Class.csv");

var newSample = new CandidateData
{
    Gender = "M",
    SecondaryEducationPercentage = 100f,
    SecondaryEducationBoard = "Central",
    HigherSecondaryEducationPercentage = 90f,
    HigherSecondaryEducationBoard = "Other",
    HigherSecondaryEducationSpecialization = "Science",
    DegreePercentage = 50f,
    DegreeType = "Sci&Tech",
    WorkExperience = "No",
    EmployabilityTestPercentage = 85f,
    Specialization = "Mkt&Fin",
    MbaPercentage = 66.28f
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

    trainer.Fit(dataFile);

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
    Console.WriteLine($"Score: {prediction.Score:#.##}");
    Console.WriteLine("------------------------------");
});