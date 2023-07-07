using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace SciSharp.Models.Transformer
{
    public class TransformerClassificationConfig
    {
        public DatasetConfig DatasetCfg;
        public ModelConfig ModelCfg;
        public TrainConfig TrainCfg;

        public TransformerClassificationConfig()
        {
            DatasetCfg = new DatasetConfig();
            ModelCfg = new ModelConfig();
            TrainCfg = new TrainConfig();
        }

        public class DatasetConfig
        {
            public int vocab_size = 20000; // Only consider the top 20k words
            public int maxlen = 200; // Only consider the first 200 words of each movie review
        }

        public class ModelConfig
        {
            public int embed_dim = 32; // Embedding size for each token
            public int num_heads = 2;  // Number of attention heads
            public int ff_dim = 32;    // Hidden layer size in feed forward network inside transformer
            public float dropout_rate = 0.1f; // Dropout rate
            public int dense_dim = 20;
        }

        public class TrainConfig
        {
            public int batch_size = 32;
            public int epochs = 20;
        }
    }
}
