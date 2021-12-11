using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models.ObjectDetection
{
    public class YoloTrainingOptions : TrainingOptions
    {
        public new YoloDataset TrainingData { get; set; }
        public new YoloDataset TestingData { get; set; }
    }
}
