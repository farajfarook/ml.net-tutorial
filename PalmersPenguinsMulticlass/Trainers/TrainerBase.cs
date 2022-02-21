using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using PalmersPenguinsMulticlass.Models;

namespace PalmersPenguinsMulticlass.Trainers;

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
    protected ITrainerEstimator<MulticlassPredictionTransformer<TParameters>, TParameters> Model;
    private ITransformer _trainedModel;


    public void Fit(string trainingFileName)
    {
        if (!File.Exists(trainingFileName))
        {
            throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
        }

        _dataSplit = LoadAndPrepareData(trainingFileName);
        var dataProcessPipeline = BuildDataProcessingPipeline();
        var trainingPipeline = 
            dataProcessPipeline
                .Append(Model)
                .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));;

        _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
    }

    private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
    {
        var trainingDataView = MlContext.Data
            .LoadFromTextFile<PalmerPenguinsData>
                (trainingFileName, hasHeader: true, separatorChar: ',');
        return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
    }

    public MulticlassClassificationMetrics  Evaluate()
    {
        var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);
        return MlContext.MulticlassClassification.Evaluate(testSetTransform);
    }

    public void Save()
    {
        MlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
    }
    
    private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
    {
        var dataProcessPipeline = MlContext.Transforms.Conversion
            .MapValueToKey(inputColumnName: nameof(PalmerPenguinsData.Label), outputColumnName: "Label")
            
            .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sex", outputColumnName: "SexFeaturized"))
            .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Island", outputColumnName: "IslandFeaturized"))
            
            .Append(MlContext.Transforms.Concatenate("Features",
                "IslandFeaturized",
                nameof(PalmerPenguinsData.CulmenLength),
                nameof(PalmerPenguinsData.CulmenDepth),
                nameof(PalmerPenguinsData.BodyMass),
                nameof(PalmerPenguinsData.FliperLength),
                "SexFeaturized"
            ))
            .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
            
            .AppendCacheCheckpoint(MlContext);

        return dataProcessPipeline;
    }
}