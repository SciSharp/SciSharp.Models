using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.Keras.ArgsDefinition;

namespace SciSharp.Models.TensorflowHub;

public class KerasLayerArgs : LayerArgs
{
    public string HandleName { get; set; }
}
