using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class IMDbDataset
    {
        TransformerClassificationConfig cfg;

        public IMDbDataset()
        {
            cfg = new TransformerClassificationConfig();
        }
        public IMDbDataset(TransformerClassificationConfig cfg)
        {
            this.cfg = cfg;
        }
        public IMDbDataset(int vocab_size, int maxlen)
        {
            cfg = new TransformerClassificationConfig();
            cfg.DatasetCfg.vocab_size = vocab_size;
            cfg.DatasetCfg.maxlen = maxlen;
        }

        public Tensor[] GetData()
        {
            var dataset = keras.datasets.imdb.load_data(num_words: cfg.DatasetCfg.vocab_size);
            var x_train = dataset.Train.Item1;
            var y_train = dataset.Train.Item2;
            var x_val = dataset.Test.Item1;
            var y_val = dataset.Test.Item2;

            x_train = keras.preprocessing.sequence.pad_sequences(RemoveZeros(x_train), maxlen: cfg.DatasetCfg.maxlen);
            x_val = keras.preprocessing.sequence.pad_sequences(RemoveZeros(x_val), maxlen: cfg.DatasetCfg.maxlen);
            print(len(x_train) + " Training sequences");
            print(len(x_val) + " Validation sequences");

            return new[] { x_train.astype(np.float32), y_train.astype(np.float32), x_val.astype(np.float32), y_val.astype(np.float32) };
        }

        IEnumerable<int[]> RemoveZeros(NDArray data)
        {
            var data_array = (int[,])data.ToMultiDimArray<int>();
            List<int[]> new_data = new List<int[]>();
            for (var i = 0; i < data_array.GetLength(0); i++)
            {
                List<int> new_array = new List<int>();
                for (var j = 0; j < data_array.GetLength(1); j++)
                {
                    if (data_array[i, j] == 0)
                        break;
                    else
                        new_array.Add(data_array[i, j]);
                }
                new_data.Add(new_array.ToArray());
            }
            return new_data;
        }
    }
}
