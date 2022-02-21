using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Placement.Models;

namespace Placement.Trainers;

public abstract class TrainerBase<TParameters> : ITrainerBase
    where TParameters : class
{
    protected TrainerBase(string name)
    {
        Name = name;
        MlContext = new MLContext(111);
    }

    public string Name { get; }

    private string ModelPath =>
        Path.Combine(AppContext.BaseDirectory, $"model_{Name.Replace(" ", "").ToLower()}.mdl");

    protected readonly MLContext MlContext;

    private DataOperationsCatalog.TrainTestData _dataSplit;
    protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> Model;
    private ITransformer _trainedModel;


    public void Fit(string trainingFileName)
    {
        if (!File.Exists(trainingFileName))
        {
            throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
        }

        _dataSplit = LoadAndPrepareData(trainingFileName);
        var dataProcessPipeline = BuildDataProcessingPipeline();
        var trainingPipeline = dataProcessPipeline.Append(Model);

        _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
    }

    private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
    {
        var trainingDataView = MlContext.Data
            .LoadFromTextFile<CandidateData>
                (trainingFileName, hasHeader: true, separatorChar: ',');
        return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
    }

    public BinaryClassificationMetrics  Evaluate()
    {
        var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);
        return MlContext.BinaryClassification.EvaluateNonCalibrated(testSetTransform);
    }

    public void Save()
    {
        MlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
    }
    
    private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
    {
        var tf = MlContext.Transforms;

        TextFeaturizingEstimator FeatText(string name) => tf.Text.FeaturizeText(name, name);

        var statusMap = new Dictionary<string, bool> { { "Placed", true }, { "Not Placed", false } };
        
        var dataProcessPipeline = tf
            .Conversion.MapValue("Label", statusMap, nameof(CandidateData.Status))
    
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
            .AppendCacheCheckpoint(MlContext);

        return dataProcessPipeline;
    }
}