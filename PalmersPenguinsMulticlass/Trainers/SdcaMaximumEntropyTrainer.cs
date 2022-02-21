using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PalmersPenguinsMulticlass.Trainers;

public class SdcaMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
{
    public SdcaMaximumEntropyTrainer() : base("Sdca Maximum Entropy")
    {
        Model = MlContext.MulticlassClassification.Trainers
            .SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
    }
}