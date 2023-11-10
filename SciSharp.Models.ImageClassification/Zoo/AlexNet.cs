using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.ImageClassification.Zoo
{
    public class AlexNet : IModelZoo
    {
        public IModel BuildModel(FolderClassificationConfig config)
        {
            var model = keras.Sequential(new List<ILayer> {
                // 这里使用一个11*11的更大窗口来捕捉对象。
                // 同时，步幅为4，以减少输出的高度和宽度。
                // 另外，输出通道的数目远大于LeNet
                keras.layers.Conv2D(filters:96, kernel_size: 11, strides: 4, activation:"relu"),
                keras.layers.MaxPooling2D(pool_size: 3, strides: 2),
                // 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
                keras.layers.Conv2D(filters:256, kernel_size: 5, padding:"same", activation:"relu"),
                keras.layers.MaxPooling2D(pool_size: 3, strides: 2),
                //  使用三个连续的卷积层和较小的卷积窗口。
                //  除了最后的卷积层，输出通道的数量进一步增加。
                //  在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
                keras.layers.Conv2D(filters:384, kernel_size: 3, padding:"same", activation:"relu"),
                keras.layers.Conv2D(filters:384, kernel_size: 3, padding:"same", activation:"relu"),
                keras.layers.Conv2D(filters:256, kernel_size: 3, padding:"same", activation:"relu"),
                keras.layers.MaxPooling2D(pool_size: 3, strides: 2),
                keras.layers.Flatten(),
                // 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
                keras.layers.Dense(4096, activation:"relu"),
                keras.layers.Dropout(0.5f),
                keras.layers.Dense(4096, activation:"relu"),
                //keras.layers.Dropout(0.2f),

                keras.layers.Dense(config.NumberOfClass, "softmax"),
            });

            var X = tf.random.normal((1, config.InputShape[0], config.InputShape[1], 3));
            model.Apply(X); // 需要走一遍

            var optimizer = keras.optimizers.SGD(0.0001f);
            //var optimizer = keras.optimizers.Adam();
            var loss = keras.losses.SparseCategoricalCrossentropy();
            model.compile(optimizer, loss, new[] { "accuracy" });

            return model;
        }
    }
}
