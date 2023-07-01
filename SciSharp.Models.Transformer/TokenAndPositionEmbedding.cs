using Serilog.Debugging;
using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class TokenAndPositionEmbedding
    {
        ILayer token_emb;
        ILayer pos_emb;

        public TokenAndPositionEmbedding(int maxlen, int vocab_size, int embed_dim)
        {
            token_emb = keras.layers.Embedding(input_dim: vocab_size, output_dim: embed_dim);
            pos_emb = keras.layers.Embedding(input_dim: maxlen, output_dim: embed_dim);
        }

        public Tensor Apply(Tensor x)
        {
            //var maxlen = tf.shape(x)[-1];
            //var positions = tf.range(start: 0, limit: maxlen, delta: 1);
            //positions = pos_emb.Apply(positions);
            //var embedding = token_emb.Apply(x);
            //var output = embedding + positions;
            return token_emb.Apply(x);
        }
    }
}
