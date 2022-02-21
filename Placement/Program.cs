// See https://aka.ms/new-console-template for more information

using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using Placement;
using Placement.Models;

var trainingFileName = Path.Combine(AppContext.BaseDirectory, "Data", "Placement_Data_Full_Class.csv");
var mlContext = new MLContext(111);

var trainingDataView = mlContext.Data
    .LoadFromTextFile<CandidateData>
        (trainingFileName, hasHeader: true, separatorChar: ',');

var dataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);

// DATA TRANSFORMATION

var tf = mlContext.Transforms;

TextFeaturizingEstimator FeatText(string name) => tf.Text.FeaturizeText(name, name);

var dataProcessPipeline = tf
    .Conversion.MapValueToKey(inputColumnName: nameof(CandidateData.Status), outputColumnName: "Label")
    
    .Append(FeatText(nameof(CandidateData.Gender)))
    .Append(FeatText(nameof(CandidateData.Specialization)))
    .Append(FeatText(nameof(CandidateData.DegreeType)))
    .Append(FeatText(nameof(CandidateData.WorkExperience)))
    .Append(FeatText(nameof(CandidateData.SecondaryEducationBoard)))
    .Append(FeatText(nameof(CandidateData.HigherSecondaryEducationBoard)))
    .Append(FeatText(nameof(CandidateData.HigherSecondaryEducationSpecialization)))

    .Append(tf.Concatenate("Features",
        nameof(CandidateData.Gender),
        nameof(CandidateData.Specialization),
        nameof(CandidateData.DegreePercentage),
        nameof(CandidateData.DegreeType),
        nameof(CandidateData.MbaPercentage),
        nameof(CandidateData.WorkExperience),
        nameof(CandidateData.EmployabilityTestPercentage),
        nameof(CandidateData.SecondaryEducationBoard),
        nameof(CandidateData.SecondaryEducationPercentage),
        nameof(CandidateData.HigherSecondaryEducationBoard),
        nameof(CandidateData.HigherSecondaryEducationPercentage),
        nameof(CandidateData.HigherSecondaryEducationSpecialization)
    ))
    .Append(tf.NormalizeMinMax("Features", "Features"))
    .AppendCacheCheckpoint(mlContext);

// MODEL
var model = mlContext.MulticlassClassification.Trainers
    .LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");

var trainingPipeline = dataProcessPipeline.Append(model);
var trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

var testSetTransform = trainedModel.Transform(dataSplit.TestSet);
var modelMetrics = mlContext.MulticlassClassification.Evaluate(testSetTransform);

Console.WriteLine($"Macro Accuracy: {modelMetrics.MacroAccuracy:#.##}{Environment.NewLine}" +
                  $"Micro Accuracy: {modelMetrics.MicroAccuracy:#.##}{Environment.NewLine}" +
                  $"Log Loss: {modelMetrics.LogLoss:#.##}{Environment.NewLine}" +
                  $"Log Loss Reduction: {modelMetrics.LogLossReduction:#.##}{Environment.NewLine}");
