using Microsoft.ML;
using PalmerPenguins.Models;

namespace PalmerPenguins;

public class Predictor
{
    private readonly string _name;
    private string ModelPath => Path.Combine(AppContext.BaseDirectory, $"model_{_name.Replace(" ", "")}.mdl");
    private readonly MLContext _mlContext;

    public Predictor(string name)
    {
        _name = name;
        _mlContext = new MLContext(111);
    }

    public PalmerPenguinsBinaryPrediction Predict(PalmerPenguinsBinaryData newSample)
    {
        var model = LoadModel();
        var predictionEngine =
            _mlContext.Model.CreatePredictionEngine<PalmerPenguinsBinaryData, PalmerPenguinsBinaryPrediction>(model);

        return predictionEngine.Predict(newSample);
    }
    
    private ITransformer LoadModel()
    {
        if (!File.Exists(ModelPath)) 
            throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
        using var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        return _mlContext.Model.Load(stream, out _) ?? throw new Exception($"Failed to load Model");
    }
}