using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PalmerPenguins.Trainers;

public class PriorTrainer : 
    TrainerBase<PriorModelParameters>
{
    public PriorTrainer() : base("Prior")
    {
        Model = MlContext
            .BinaryClassification
            .Trainers
            .Prior(labelColumnName: "Label");
    }
}