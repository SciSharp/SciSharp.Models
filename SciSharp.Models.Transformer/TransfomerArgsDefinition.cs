using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;

namespace Tensorflow.Keras.ArgsDefinition
{
    public class TokenAndPositionEmbeddingArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("max_len")]
        public int Maxlen { get; set; }
        [JsonProperty("vocab_sise")]
        public int VocabSize { get; set; }
        [JsonProperty("embed_dim")]
        public int EmbedDim { get; set; }
        [JsonProperty("activity_regularizer")]
        public override IRegularizer ActivityRegularizer { get => base.ActivityRegularizer; set => base.ActivityRegularizer = value; }
    }
    public class TransformerBlockArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("embed_dim")]
        public int EmbedDim { get; set; }
        [JsonProperty("num_heads")]
        public int NumHeads { get; set; }
        [JsonProperty("ff_dim")]
        public int FfDim { get; set; }
        [JsonProperty("dropout_rate")]
        public float DropoutRate { get; set; } = 0.1f;
        [JsonProperty("activity_regularizer")]
        public override IRegularizer ActivityRegularizer { get => base.ActivityRegularizer; set => base.ActivityRegularizer = value; }
    }
    public class TransformerClassificationArgs : AutoSerializeLayerArgs
    {
        [JsonProperty("max_len")]
        public int Maxlen { get; set; }
        [JsonProperty("vocab_sise")]
        public int VocabSize { get; set; }
        [JsonProperty("embed_dim")]
        public int EmbedDim { get; set; }
        [JsonProperty("num_heads")]
        public int NumHeads { get; set; }
        [JsonProperty("ff_dim")]
        public int FfDim { get; set; }
        [JsonProperty("dropout_rate")]
        public float DropoutRate { get; set; } = 0.1f;
        [JsonProperty("dense_dim")]
        public int DenseDim { get; set; }
        [JsonProperty("activity_regularizer")]
        public override IRegularizer ActivityRegularizer { get => base.ActivityRegularizer; set => base.ActivityRegularizer = value; }
    }
}
