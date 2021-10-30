using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models
{
    public class ModelTestResult
    {
        public float Loss { get; set; }
        public float Accuracy { get; set; }
        public NDArray Predictions { get; set; }
    }
}
