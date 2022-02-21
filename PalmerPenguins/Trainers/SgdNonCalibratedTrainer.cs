using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PalmerPenguins.Trainers;

public class SgdNonCalibratedTrainer : TrainerBase<LinearBinaryModelParameters>
{
    public SgdNonCalibratedTrainer() : base("Sgd NonCalibrated")
    {
        Model = MlContext
            .BinaryClassification
            .Trainers
            .SgdNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
    }
}