using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class TransformerBlock
    {
        ILayer att;
        ILayer dropout1;
        ILayer add1;
        ILayer layernorm1;
        ILayer ffn1;
        ILayer ffn2;
        ILayer dropout2;
        ILayer add2;
        ILayer layernorm2;  

        public TransformerBlock(int embed_dim, int num_heads, int ff_dim, float dropout_rate = 0.1f)
        {
            att = keras.layers.MultiHeadAttention(num_heads, embed_dim);
            dropout1 = keras.layers.Dropout(dropout_rate);
            add1 = keras.layers.Add();
            layernorm1 = keras.layers.LayerNormalization(axis: null, epsilon: 1e-6f);
            ffn1 = keras.layers.Dense(ff_dim, activation: "relu");
            ffn2 = keras.layers.Dense(embed_dim);
            dropout2 = keras.layers.Dropout(dropout_rate);
            add2 = keras.layers.Add();
            layernorm2 = keras.layers.LayerNormalization(axis: null, epsilon: 1e-6f);           
        }
        public Tensor Apply(Tensor inputs, bool training = true)
        {
            var att_output = att.Apply(new Tensors(inputs, inputs));
            att_output = dropout1.Apply(att_output, training: training);
            var residual = add1.Apply(new Tensors(inputs, att_output));
            var out1 = layernorm1.Apply(residual);
            var ffn_output = ffn1.Apply(out1);
            ffn_output = ffn2.Apply(ffn_output);
            ffn_output = dropout2.Apply(ffn_output, training: training);
            var output = add2.Apply(new Tensors(out1, ffn_output));
            output = layernorm2.Apply(output);
            return output;
        }
    }
}
