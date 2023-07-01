using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class TransformerDataset
    {
        TransformerConfig cfg;

        public TransformerDataset()
        {
            cfg = new TransformerConfig();
        }

        public TransformerDataset(TransformerConfig config)
        {
            cfg = config;
        }

        public TransformerDataset(int vocab_size, int maxlen)
        {
            cfg = new TransformerConfig();
            cfg.DatasetConfig.vocab_size = vocab_size;
            cfg.DatasetConfig.maxlen = maxlen;
        }

        public Tensor[] GetData()
        {
            var dataset = keras.datasets.imdb.load_data(maxlen: cfg.DatasetConfig.maxlen);      
            var x_train = dataset.Train.Item1.astype(np.float32);
            var y_train = dataset.Train.Item2.astype(np.float32);
            var x_val = dataset.Test.Item1.astype(np.float32);
            var y_val = dataset.Test.Item2.astype(np.float32);
            print(len(x_train) + "Training sequences");
            print(len(x_val) + "Validation sequences");
            return new[] { x_train, y_train, x_val, y_val };
        }
    }
}
