using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class TransformerBlock
    {
        ILayer att;
        ILayer ffn;
        ILayer layernorm1;
        ILayer layernorm2;
        ILayer dropout1;
        ILayer dropout2;

        public TransformerBlock(int embed_dim, int num_heads, int ff_dim, float rate = 0.1f)
        {
            att = keras.layers.MultiHeadAttention(num_heads, embed_dim);
            ffn = keras.Sequential(new List<ILayer> { keras.layers.Dense(ff_dim, activation: keras.activations.Relu), keras.layers.Dense(embed_dim) });
            layernorm1 = keras.layers.LayerNormalization(axis: null, epsilon: 1e-6f);
            layernorm2 = keras.layers.LayerNormalization(axis: null, epsilon: 1e-6f);
            dropout1 = keras.layers.Dropout(rate);
            dropout2 = keras.layers.Dropout(rate);
        }
        public Tensor Apply(Tensor inputs, bool training)
        {
            var att_output = att.Apply(inputs, inputs);
            att_output = dropout1.Apply(att_output, training: training);
            var out1 = layernorm1.Apply(inputs + att_output);
            var ffn_output = ffn.Apply(out1);
            ffn_output = dropout2.Apply(ffn_output, training: training);
            return layernorm2.Apply((Tensor)out1 + (Tensor)ffn_output);
        }
    }
}
