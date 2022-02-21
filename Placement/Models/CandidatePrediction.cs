using Microsoft.ML.Data;

namespace Placement.Models;

public class CandidatePrediction
{
    [ColumnName("PredictedLabel")]
    public string  PredictedLabel { get; set; }
}