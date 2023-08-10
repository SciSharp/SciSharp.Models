using Serilog.Debugging;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    public class TokenAndPositionEmbedding : Layer
    {
        TokenAndPositionEmbeddingArgs args;
        Tensor positions_base;
        ILayer token_emb;
        ILayer pos_emb;

        public TokenAndPositionEmbedding(TokenAndPositionEmbeddingArgs args) : base(args)
        {
            this.args = args;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
            positions_base = tf.constant(tf.range(start: 0, limit: args.Maxlen, delta: 1));
            token_emb = keras.layers.Embedding(input_dim: args.VocabSize, output_dim: args.EmbedDim);
            pos_emb = keras.layers.Embedding(input_dim: args.Maxlen, output_dim: args.EmbedDim);
            StackLayers(token_emb, pos_emb);
            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = false, IOptionalArgs? optional_args = null)
        {
            var embedding = token_emb.Apply(inputs, state, training ?? false, optional_args);
            var positions = pos_emb.Apply(positions_base, state, training ?? false, optional_args);
            return (Tensor)embedding + (Tensor)positions;
        }

        public Tensors ComputeMask(Tensors inputs, Tensor? mask = null)
        {
            mask = mask ?? new Tensors(0);
            return math_ops.not_equal(inputs, mask);
        }
    }
}
