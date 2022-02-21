using Microsoft.ML.Data;

namespace PalmerPenguins.Trainers;

public interface ITrainerBase
{
    string Name { get; }
    void Fit(string trainingFileName);
    BinaryClassificationMetrics Evaluate();
    void Save();
}