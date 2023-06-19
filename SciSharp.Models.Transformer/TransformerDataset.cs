using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.Transformer
{
    public class TransformerDataset
    {
        public Tensor[] GetData()
        {
            var vocab_size = 20000; // Only consider the top 20k words
            var maxlen = 200; // Only consider the first 200 words of each movie review
            var dataset = keras.datasets.imdb.load_data(num_words: vocab_size);
            var x_train = dataset.Train.Item1;
            var y_train = dataset.Train.Item2;
            var x_val = dataset.Test.Item1;
            var y_val = dataset.Test.Item2;
            print(len(x_train) + "Training sequences");
            print(len(x_val) + "Validation sequences");
            x_train = keras.preprocessing.sequence.pad_sequences((IEnumerable<int[]>)x_train, maxlen: maxlen);
            x_val = keras.preprocessing.sequence.pad_sequences((IEnumerable<int[]>)x_val, maxlen: maxlen);
            return new[] { x_train, y_train, x_val, y_val };
        }
    }
}
