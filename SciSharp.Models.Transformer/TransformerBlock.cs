﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    public class TransformerBlock : Layer
    {
        TransformerBlockArgs args;
        ILayer att;
        ILayer dropout1;
        ILayer layernorm1;
        ILayer ffn1;
        ILayer ffn2;
        ILayer dropout2;
        ILayer layernorm2;

        public TransformerBlock(TransformerBlockArgs args) : base(args)
        {
            this.args = args;
        }
        public override void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
            att = keras.layers.MultiHeadAttention(args.NumHeads, args.EmbedDim);
            dropout1 = keras.layers.Dropout(args.DropoutRate);
            layernorm1 = keras.layers.LayerNormalization(axis: -1, epsilon: 1e-6f);
            ffn1 = keras.layers.Dense(args.FfDim, activation: "relu");
            ffn2 = keras.layers.Dense(args.EmbedDim);
            dropout2 = keras.layers.Dropout(args.DropoutRate);
            layernorm2 = keras.layers.LayerNormalization(axis: -1, epsilon: 1e-6f);
            StackLayers(att, dropout1, layernorm1, ffn1, ffn2, dropout2, layernorm2);
            built = true;
        }
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var att_output = att.Apply(new Tensors(inputs, inputs), state, training, optional_args);
            att_output = dropout1.Apply(att_output, state, training, optional_args);
            var out1 = layernorm1.Apply((Tensor)inputs + (Tensor)att_output, state, training, optional_args);
            var ffn_output = ffn1.Apply(out1, state, training, optional_args);
            ffn_output = ffn2.Apply(ffn_output, state, training, optional_args);
            ffn_output = dropout2.Apply(ffn_output, state, training, optional_args);
            var output = layernorm2.Apply((Tensor)out1 + (Tensor)ffn_output, state, training, optional_args);
            return output;
        }
    }
}
