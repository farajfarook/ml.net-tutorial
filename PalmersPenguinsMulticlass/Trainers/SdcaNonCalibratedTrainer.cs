using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PalmersPenguinsMulticlass.Trainers;

public class SdcaNonCalibratedTrainer : TrainerBase<LinearMulticlassModelParameters>
{
    public SdcaNonCalibratedTrainer() : base("Sdca NonCalibrated")
    {
        Model = MlContext.MulticlassClassification.Trainers
            .SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
    }
}