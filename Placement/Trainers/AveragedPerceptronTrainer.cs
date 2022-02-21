using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace Placement.Trainers;

public class AveragedPerceptronTrainer : 
    TrainerBase<LinearBinaryModelParameters>
{
    public AveragedPerceptronTrainer() : base("Averaged Perceptron")
    {
        Model = MlContext
            .BinaryClassification
            .Trainers
            .AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features");
    }
}