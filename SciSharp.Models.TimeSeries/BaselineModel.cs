using System;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.TimeSeries
{
    public class BaselineModel : ModelBase, ITimeSeriesTask
    {
        public new void Train(TrainingOptions options)
        {
            var baseline = new Baseline(_window.GetColumnIndex("T (degC)"));
            baseline.compile(loss: keras.losses.MeanSquaredError(),
                          optimizer: keras.optimizers.Adam(),
                          metrics: new[] { "mae" });

            baseline.evaluate(options.Dataset.Item2);
        }
    }
}
