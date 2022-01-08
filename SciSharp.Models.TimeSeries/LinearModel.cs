using System;
using System.Collections.Generic;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.TimeSeries
{
    public class LinearModel : ModelBase, ITimeSeriesTask
    {
        protected override Model BuildModel()
        {
            var model = keras.Sequential(new List<ILayer> { keras.layers.Dense(units: 1) });

            /*early_stopping = keras.callbacks.EarlyStopping(monitor = "val_loss",
                                                  patience = patience,
                                                  mode = 'min')*/

            model.compile(loss: keras.losses.MeanSquaredError(),
                          optimizer: keras.optimizers.Adam(),
                          metrics: new[] { "mae" });
            return model;
        }
    }
}
