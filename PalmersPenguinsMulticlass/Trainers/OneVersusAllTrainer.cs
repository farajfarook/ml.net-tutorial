using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PalmersPenguinsMulticlass.Trainers;

public class OneVersusAllTrainer : TrainerBase<OneVersusAllModelParameters>
{
    public OneVersusAllTrainer() : base("One Versus All")
    {
        Model = MlContext.MulticlassClassification.Trainers
            .OneVersusAll(binaryEstimator: MlContext.BinaryClassification.Trainers.SgdCalibrated());
    }
}