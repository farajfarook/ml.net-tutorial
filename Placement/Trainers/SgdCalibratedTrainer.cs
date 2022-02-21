using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace Placement.Trainers;

public class SgdCalibratedTrainer 
    : TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public SgdCalibratedTrainer() : base("Sgd Calibrated")
    {
        Model = MlContext
            .BinaryClassification
            .Trainers
            .SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features");
    }
}