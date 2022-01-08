using System;
using System.Collections.Generic;
using System.IO;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.TimeSeries
{
    public class ConvolutionalModel : ModelBase, ITimeSeriesTask
    {
        protected override Model BuildModel()
        {
            var model = keras.Sequential(new List<ILayer>
            {
                keras.layers.Conv1D(filters: 32, kernel_size: _args.InputWidth, activation: "relu"),
                keras.layers.Dense(units: 32, activation: "relu"),
                keras.layers.Dense(units: 1)
            });

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
