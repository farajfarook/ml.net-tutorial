using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PalmersPenguinsMulticlass.Trainers;

public class LbfgsMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
{
    public LbfgsMaximumEntropyTrainer() : base("LBFGS Maximum Entropy")
    {
        Model = MlContext.MulticlassClassification.Trainers
            .LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
    }
}