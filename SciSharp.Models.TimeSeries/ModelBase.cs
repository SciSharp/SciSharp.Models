using PandasNet;
using System;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace SciSharp.Models.TimeSeries
{
    // https://www.tensorflow.org/tutorials/structured_data/time_series
    public class ModelBase
    {
        protected WindowGenerator _window;
        protected TimeSeriesModelArgs _args;
        protected TaskOptions _taskOptions;
        protected Model model;
        public (IDatasetV2, IDatasetV2, IDatasetV2) GenerateDataset<TDataSource>(Func<TDataSource> preprocess)
        {
            var ds = preprocess();
            if (ds is DataFrame df)
            {
                _window = new WindowGenerator(input_width: _args.InputWidth, label_width: 1, shift: 1,
                    columns: df.columns,
                    label_columns: _args.LabelColumns);

                return _window.GenerateDataset(df);
            }
            else
                throw new NotImplementedException("");
        }

        public void SetModelArgs<T>(T args)
        {
            if (args is TimeSeriesModelArgs tsArgs)
                _args = tsArgs;
            else
                throw new ValueError($"Please set model args as {nameof(TimeSeriesModelArgs)}");
        }

        public void Config(TaskOptions options)
        {
            _taskOptions = options;
        }

        protected virtual Model BuildModel()
        {
            throw new NotImplementedException("");
        }

        public void Train(TrainingOptions options)
        {
            model = BuildModel();
            model.fit(options.Dataset.Item1, epochs: options.Epochs, validation_data: options.Dataset.Item2);
            SaveModel();
        }

        protected void SaveModel()
        {
            var filePath = new FileInfo(_taskOptions.WeightsPath);
            Directory.CreateDirectory(filePath.Directory.FullName);
            model.save_weights(_taskOptions.WeightsPath);
        }

        public float Test(TestingOptions options)
        {
            if (model == null)
                model = BuildModel();
            foreach (var input in options.Dataset.take(1))
                model.Apply(input);
            model.load_weights(_taskOptions.WeightsPath);
            var result = model.evaluate(options.Dataset);
            return result.First(x => x.Key == "mean_absolute_error").Value;
        }

        public Tensor Predict(Tensor input)
        {
            if (model == null)
                model = BuildModel();
            var result = model.predict(input);
            return result[0];
        }
    }
}
