using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models
{
    public class TrainingOptions
    {
        public float TestingPercentage { get; set; } = 0.1f;
        public float ValidationPercentage = 0.1f;

        public TrainingOptions()
        {

        }
    }
}
