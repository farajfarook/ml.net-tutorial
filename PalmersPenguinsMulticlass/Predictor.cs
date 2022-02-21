using Microsoft.ML;
using PalmersPenguinsMulticlass.Models;

namespace PalmerPenguins;

public class Predictor
{
    private readonly string _name;
    private string ModelPath => Path.Combine(AppContext.BaseDirectory, $"model_{_name.Replace(" ", "")}.mdl");
    private readonly MLContext _mlContext;

    private ITransformer _model;

    public Predictor(string name)
    {
        _name = name;
        _mlContext = new MLContext(111);
    }
    
    public PalmerPenguinsPrediction Predict(PalmerPenguinsData newSample)
    {
        LoadModel();

        var predictionEngine = _mlContext.Model.CreatePredictionEngine<PalmerPenguinsData, PalmerPenguinsPrediction>(_model);

        return predictionEngine.Predict(newSample);
    }

    private void LoadModel()
    {
        if (!File.Exists(ModelPath))
        {
            throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
        }

        using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            _model = _mlContext.Model.Load(stream, out _);
        }

        if (_model == null)
        {
            throw new Exception($"Failed to load Model");
        }
    }
}