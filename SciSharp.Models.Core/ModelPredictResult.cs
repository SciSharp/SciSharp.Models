using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models
{
    public class ModelPredictResult
    {
        public NDArray Predictions { get; set; }
        public NDArray Probabilities { get; set; }
    }
}
