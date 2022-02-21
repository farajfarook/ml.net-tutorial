using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace PalmerPenguins.Trainers;

public class SdcaLogisticRegressionTrainer : 
    TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
{
    public SdcaLogisticRegressionTrainer() : base("Sdca Logistic Regression")
    {
        Model = MlContext
            .BinaryClassification
            .Trainers
            .SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
    }
}