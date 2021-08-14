using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models
{
    public class TrainingOptions
    {
        public string DataDir { get; set; }
        public float TestingPercentage { get; set; } = 0.2f;
        public float ValidationPercentage { get; set; } = 0.1f;
        public string ModelPath { get; set; }
        public string LabelPath { get; set; }
        public string CheckpointPath { get; set; }
        public TrainingOptions()
        {

        }
    }
}
