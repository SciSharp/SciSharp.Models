using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models
{
    public class FeatureAndLabel
    {
        NDArray _features;
        public NDArray Features => _features;
        NDArray _labels;
        public NDArray Labels => _labels;
        public FeatureAndLabel(NDArray features, NDArray labels)
        {
            _features = features;
            _labels = labels;
        }
    }
}
