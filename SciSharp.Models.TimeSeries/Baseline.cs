using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace SciSharp.Models.TimeSeries
{
    public class Baseline : Model
    {
        int _label_index;

        public Baseline(int label_index) : base(new ModelArgs { })
        {
            _label_index = label_index;
        }

        protected override Tensors Call(Tensors inputs, Tensor state = null, bool? training = null)
        {
            var result = inputs[":", ":", $"{_label_index}"];
            return result[":", ":", tf.newaxis];
        }
    }
}
