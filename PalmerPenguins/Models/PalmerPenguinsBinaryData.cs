using Microsoft.ML.Data;

namespace PalmerPenguins.Models;

/// <summary>
/// Models Palmer Penguins Binary Data.
/// </summary>
public class PalmerPenguinsBinaryData
{
    [LoadColumn(0)]
    public bool Label { get; set; }

    [LoadColumn(1)]
    public float BillLength { get; set; }

    [LoadColumn(2)]
    public float BIllDepth { get; set; }
}