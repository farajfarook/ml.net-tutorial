using Microsoft.ML.Data;

namespace PalmersPenguinsMulticlass.Trainers;

public interface ITrainerBase
{
    string Name { get; }
    void Fit(string trainingFileName);
    MulticlassClassificationMetrics Evaluate();
    void Save();
}