using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace Placement.Trainers;

public class LbfgsLogisticRegressionTrainer: TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, 
    PlattCalibrator>>
{
    public LbfgsLogisticRegressionTrainer() : base("LBFGS Logistic Regression")
    {
        Model = MlContext.BinaryClassification
            .Trainers
            .LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
    }
}