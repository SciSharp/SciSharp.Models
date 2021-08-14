using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SciSharp.Models
{
    public class ModelOptions
    {
        public string DataDir { get; set; }
        public TrainingOptions TrainingOptions { get; set; }

        public ModelOptions()
        {
            TrainingOptions = new TrainingOptions();
        }
    }
}
