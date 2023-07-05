using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class TransformerBlockArgs : Tensorflow.Keras.ArgsDefinition.AutoSerializeLayerArgs
    {
        public int EmbedDim { get; set; }
        public int NumHeads { get; set; }
        public int FfDim { get; set; }
        public float DropoutRate { get; set; } = 0.1f;
        public override IRegularizer ActivityRegularizer { get => base.ActivityRegularizer; set => base.ActivityRegularizer = value; }
    }
    public class TransformerBlock : Layer
    {
        TransformerBlockArgs args;
        ILayer att;
        ILayer dropout1;
        ILayer add1;
        ILayer layernorm1;
        ILayer ffn1;
        ILayer ffn2;
        ILayer dropout2;
        ILayer add2;
        ILayer layernorm2;

        public TransformerBlock(TransformerBlockArgs args)
        : base(new LayerArgs
        {
            DType = args.DType,
            Name = args.Name,
            InputShape = args.InputShape,
            BatchSize = args.BatchSize
        })
        {
            this.args = args;
        }
        public override void build(KerasShapesWrapper input_shape)
        {
            att = keras.layers.MultiHeadAttention(args.NumHeads, args.EmbedDim);
            dropout1 = keras.layers.Dropout(args.DropoutRate);
            add1 = keras.layers.Add();
            layernorm1 = keras.layers.LayerNormalization(axis: null, epsilon: 1e-6f);
            ffn1 = keras.layers.Dense(args.FfDim, activation: "relu");
            ffn2 = keras.layers.Dense(args.EmbedDim);
            dropout2 = keras.layers.Dropout(args.DropoutRate);
            add2 = keras.layers.Add();
            layernorm2 = keras.layers.LayerNormalization(axis: null, epsilon: 1e-6f);
            StackLayers(att, dropout1, add1, layernorm1, ffn1, ffn2, dropout2, add2, layernorm2);
        }
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var att_output = att.Apply(new Tensors(inputs, inputs), state, training?? false, optional_args);
            att_output = dropout1.Apply(att_output, state, training?? false, optional_args);
            var residual = add1.Apply(new Tensors(inputs, att_output), state, training?? false, optional_args);
            var out1 = layernorm1.Apply(residual, state, training?? false, optional_args);
            var ffn_output = ffn1.Apply(out1, state, training?? false, optional_args);
            ffn_output = ffn2.Apply(ffn_output, state, training?? false, optional_args);
            ffn_output = dropout2.Apply(ffn_output, state, training?? false, optional_args);
            var output = add2.Apply(new Tensors(out1, ffn_output), state, training?? false, optional_args);
            output = layernorm2.Apply(output, state, training?? false, optional_args);
            return output;
        }
    }
}
