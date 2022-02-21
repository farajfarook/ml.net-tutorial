using Microsoft.ML.Data;

namespace Placement.Models;

public class CandidateData
{
    [LoadColumn(1)]
    public string Gender { get; set; }

    [LoadColumn(2)]
    public float SecondaryEducationPercentage { get; set; }

    [LoadColumn(3)]
    public string SecondaryEducationBoard { get; set; }
    
    [LoadColumn(4)]
    public float HigherSecondaryEducationPercentage { get; set; }
    
    [LoadColumn(5)]
    public string HigherSecondaryEducationBoard { get; set; }
    
    [LoadColumn(6)]
    public string HigherSecondaryEducationSpecialization { get; set; }

    [LoadColumn(7)]
    public float DegreePercentage { get; set; }
    
    [LoadColumn(8)]
    public string DegreeType { get; set; }
    
    [LoadColumn(9)]
    public string WorkExperience { get; set; }
    
    [LoadColumn(10)]
    public float EmployabilityTestPercentage { get; set; }
    
    [LoadColumn(11)]
    public string Specialization { get; set; }
    
    [LoadColumn(12)]
    public float MbaPercentage { get; set; }
    
    [LoadColumn(13)]
    public string Status { get; set; }
    
    [LoadColumn(14)]
    public int Salary { get; set; }
}