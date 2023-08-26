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
        ILayer token_emb;
        IVariableV1 position_embeddings;

        public TokenAndPositionEmbedding(TokenAndPositionEmbeddingArgs args) : base(args)
        {
            this.args = args;
        }

        public override void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
            token_emb = keras.layers.Embedding(input_dim: args.VocabSize, output_dim: args.EmbedDim);
            tf_with(ops.name_scope("position_embeddings"), scope =>
            {
                position_embeddings = add_weight(name: "position_embedding", shape: (args.Maxlen, args.EmbedDim));
            });
            StackLayers(token_emb);
            built = true;
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var embedding = token_emb.Apply(inputs, state, training, optional_args);
            var maxlen = inputs.shape[-1];
            var position_ids = tf.range(start: 0, limit: maxlen, delta: 1);
            var positions = tf.gather(position_embeddings.AsTensor(), indices: position_ids);
            return (Tensor)embedding + (Tensor)positions;
        }
    }
}
