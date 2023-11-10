using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.ImageClassification.Zoo
{
    public class NiN : IModelZoo
    {
        ILayer nin_block(int num_channels, int kernel_size, int strides, string padding)
        {
            return new BlocksLayer(new[]{
                keras.layers.Conv2D(num_channels, kernel_size: kernel_size, strides: strides, padding:padding, activation:"relu"),
                keras.layers.Conv2D(num_channels, kernel_size: 1, activation:"relu"),
                keras.layers.Conv2D(num_channels, kernel_size: 1, activation:"relu"),
            });
        }

        public IModel BuildModel(FolderClassificationConfig config)
        {
            var model = keras.Sequential(new[] {
                nin_block(96, 11, 4, "valid"),
                keras.layers.MaxPooling2D(pool_size:3, strides:2),
                nin_block(256, 5, 1, "same"),
                keras.layers.MaxPooling2D(pool_size:3, strides:2),
                nin_block(384, 3, 1, "same"),
                keras.layers.MaxPooling2D(pool_size:3, strides:2),
                keras.layers.Dropout(0.5f),
                //  标签类别数是10
                nin_block(config.NumberOfClass, 3, 1, "same"),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Reshape((1, 1, config.NumberOfClass)),
                // 将四维的输出转成二维的输出，其形状为(批量大小,10)
                keras.layers.Flatten(),
            });

            var X = tf.zeros((1, config.InputShape[0], config.InputShape[1], 3));
            model.Apply(X); // 需要走一遍

            var optimizer = keras.optimizers.SGD();
            // var optimizer = keras.optimizers.Adam();
            var loss = keras.losses.SparseCategoricalCrossentropy(from_logits: true);
            model.compile(optimizer, loss, new[] { "accuracy" });

            return model;
        }
    }
}
