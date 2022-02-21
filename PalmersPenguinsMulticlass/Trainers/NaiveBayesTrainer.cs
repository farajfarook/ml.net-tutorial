using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PalmersPenguinsMulticlass.Trainers;

public class NaiveBayesTrainer : TrainerBase<NaiveBayesMulticlassModelParameters>
{
    public NaiveBayesTrainer() : base("Naive Bayes")
    {
        Model = MlContext.MulticlassClassification.Trainers
            .NaiveBayes(labelColumnName: "Label", featureColumnName: "Features");
    }
}