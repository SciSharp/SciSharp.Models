using System;
using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;

namespace SciSharp.Models.TimeSeries
{
    class Baseline : Model
    {
        int _label_index;

        public Baseline(int label_index) : base(new ModelArgs { })
        {
            _label_index = label_index;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var result = inputs[":", ":", $"{_label_index}"];
            return result[new Slice(":"), new Slice(":"), tf.newaxis];
        }
    }
}
