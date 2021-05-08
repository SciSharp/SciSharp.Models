using System;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using PandasNet;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static PandasNet.PandasApi;

namespace SciSharp.Models.TimeSeries
{
    public class WindowGenerator
    {
        int _input_width;
        int _label_width;
        int _shift;
        DataFrame _train_df;
        DataFrame _val_df;
        DataFrame _test_df;
        string[] _label_columns;
        (string, int)[] _label_columns_indices;
        Dictionary<string, int> _column_indices = new Dictionary<string, int>();
        int _total_window_size;
        public int total_window_size => _total_window_size;
        Slice _input_slice;
        int[] _input_indices;
        int _label_start;
        Slice _labels_slice;
        int[] _label_indices;

        public WindowGenerator(int input_width, int label_width, int shift, 
            DataFrame train_df = null, DataFrame val_df = null, DataFrame test_df = null,
            string[] label_columns = null)
        {
            _input_width = input_width;
            _label_width = label_width;
            _shift = shift;
            _train_df = train_df;
            _val_df = val_df;
            _test_df = test_df;
            _label_columns = label_columns;

            // Work out the label column indices.
            _label_columns_indices = Enumerable.Range(0, label_columns.Length)
                .Select(x => (label_columns[x], x))
                .ToArray();

            Enumerable.Range(0, train_df.columns.Count)
                .ToList()
                .ForEach(x => _column_indices[train_df.columns[x].Name] = x);

            _total_window_size = input_width + shift;

            _input_slice = pd.slice(0, input_width);
            _input_indices = Enumerable.Range(0, _total_window_size)
                .Skip(_input_slice.Start)
                .Take(_input_slice.Stop.Value - _input_slice.Start)
                .ToArray();

            _label_start = _total_window_size - label_width;
            _labels_slice = pd.slice(_label_start);
            _label_indices = Enumerable.Range(0, _total_window_size)
                .Skip(_labels_slice.Start)
                .ToArray();
        }

        public (Tensor, Tensor) SplitWindow(Tensor features)
        {
            var inputs = features[":", _input_slice.ToString(), ":"];
            var labels = features[":", _labels_slice.ToString(), ":"];

            if(_label_columns != null)
            {
                labels = tf.stack(_label_columns
                    .Select(x => labels[":", ":", _column_indices[x].ToString()])
                    .ToArray(), axis: -1);
            }

            return (inputs, labels);
        }

        DatasetV2 MakeDataset(DataFrame df)
        {
            var data = tf.convert_to_tensor(pd.array<double, float>(df));
            var ds = keras.preprocessing.timeseries_dataset_from_array(data,
                sequence_length: _total_window_size,
                sequence_stride: 1,
                shuffle: true,
                batch_size: 32);
            throw new NotImplementedException("");
        }

        public DatasetV2 GetTrainingDataset()
             => MakeDataset(_train_df);

        public override string ToString()
        {
            return $"Total window size: {_total_window_size}\r\n" +
                $"Input indices: {_input_indices}\r\n" +
                $"Label indices: {_label_indices}\r\n" +
                $"Label column name(s): {_label_columns}";
        }
    }
}
