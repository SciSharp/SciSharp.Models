using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace SciSharp.Models
{
    public class GraphBuiltResult
    {
        public Graph Graph { get; set; }
        public Tensor Features { get; set; }
        public Tensor Labels { get; set; }
        public Tensor Loss { get; set; }
        public Tensor Accuracy { get; set; }
        public Tensor Prediction { get; set; }
        public Operation Optimizer { get; set; }
    }
}
