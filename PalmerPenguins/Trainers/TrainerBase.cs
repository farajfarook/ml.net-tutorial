using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using PalmerPenguins.Models;

namespace PalmerPenguins.Trainers;

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
            .LoadFromTextFile<PalmerPenguinsBinaryData>
                (trainingFileName, hasHeader: true, separatorChar: ',');
        return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
    }

    public BinaryClassificationMetrics Evaluate()
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
        var dataProcessPipeline = MlContext.Transforms.Concatenate("Features",
                nameof(PalmerPenguinsBinaryData.BIllDepth),
                nameof(PalmerPenguinsBinaryData.BillLength)
            )
            .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
            .AppendCacheCheckpoint(MlContext);

        return dataProcessPipeline;
    }
}