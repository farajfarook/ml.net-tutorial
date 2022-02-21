using Microsoft.ML.Data;

namespace PalmersPenguinsMulticlass.Models;

public class PalmerPenguinsPrediction
{
    [ColumnName("PredictedLabel")]
    public string  PredictedLabel { get; set; }
}