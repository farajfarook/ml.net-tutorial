using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace Placement.Trainers;

public class SdcaNonCalibratedTrainer : 
    TrainerBase<LinearBinaryModelParameters>
{
    public SdcaNonCalibratedTrainer() : base("Sdca NonCalibrated")
    {
        Model = MlContext
            .BinaryClassification
            .Trainers
            .SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
    }
}