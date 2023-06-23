using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Datasets;
using Tensorflow.NumPy;
using Xla;
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
            // var dataset = keras.datasets.imdb.load_data(num_words: vocab_size);
            var dataset = LoadData(num_words: vocab_size);
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
        // Temp Function: wait for the new version of Tensorflow.NET that fix the dtype convert bugs.
        private DatasetPass LoadData(int num_words)
        {
            var dst = Path.Combine(Path.GetTempPath(), "imdb");
            var lines = File.ReadAllLines(Path.Combine(dst, "imdb_train.txt"));
            var x_train_string = new string[lines.Length];
            var y_train = np.zeros(new int[] { lines.Length }, np.int32);
            for (int i = 0; i < lines.Length; i++)
            {
                y_train[i] = new NDArray(new[] { long.Parse(lines[i].Substring(0, 1)) }, np.int32);
                x_train_string[i] = lines[i].Substring(2);
            }

            var x_train = np.array(x_train_string);

            File.ReadAllLines(Path.Combine(dst, "imdb_test.txt"));
            var x_test_string = new string[lines.Length];
            var y_test = np.zeros(new int[] { lines.Length }, np.int32);
            for (int i = 0; i < lines.Length; i++)
            {
                y_test[i] = new NDArray(new[] { long.Parse(lines[i].Substring(0, 1)) }, np.int32);
                x_test_string[i] = lines[i].Substring(2);
            }

            var x_test = np.array(x_test_string);

            return new DatasetPass
            {
                Train = (x_train, y_train),
                Test = (x_test, y_test)
            };

        }
    }
}
