using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models
{
    public class TrainingOptions
    {
        /// <summary>
        /// Training data
        /// </summary>
        public FeatureAndLabel TrainingData { get; set; }
        /// <summary>
        /// Validation data
        /// </summary>
        public FeatureAndLabel ValidationData { get; set; }
        /// <summary>
        /// Testing data
        /// </summary>
        public FeatureAndLabel TestingData { get; set; }
        public int Epochs { get; set; } = 5;
        public int BatchSize { get; set; } = 100;
        public int TrainingSteps { get; set; } = 100;
        public float LearningRate { get; set; } = 0.001f;    

        public TrainingOptions()
        {

        }
    }
}
