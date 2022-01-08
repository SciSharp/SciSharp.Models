using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models
{
    public class ModelPredictResult
    {
        public string Label { get; set; }
        public NDArray Values { get; set; }
        public float Probability { get; set; }

        public override string ToString()
            => $"{Label}: {Probability}";
    }
}
