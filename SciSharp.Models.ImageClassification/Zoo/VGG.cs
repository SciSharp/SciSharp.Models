using System.Collections.Generic;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.ImageClassification.Zoo
{
    public class VGG : IModelZoo
    {
        ILayer vgg_block(int num_convs, int num_channels)
        {
            List<ILayer> blks = new List<ILayer>();

            for (var i = 0; i < num_convs; i++)
            {
                blks.Add(keras.layers.Conv2D(filters: num_channels, kernel_size: 3, padding: "same", activation: "relu"));
            }

            blks.add(keras.layers.MaxPooling2D(pool_size: 2, strides: 2));

            return new BlocksLayer(blks);
        }

        Sequential vgg((int, int)[] conv_arch, int classNum)
        {
            List<ILayer> layers = new List<ILayer>();

            foreach (var (num_convs, num_channels) in conv_arch)
            {
                layers.add(vgg_block(num_convs, num_channels));
            }

            layers.AddRange(new[] {
                keras.layers.Flatten(),
                keras.layers.Dense(4096, activation:"relu"),
                keras.layers.Dropout(0.25f),
                keras.layers.Dense(4096, activation: "relu"),
                 // keras.layers.Dropout(0.25f),
                keras.layers.Dense(classNum, activation:"softmax"),
            });

            return keras.Sequential(layers);
        }

        public IModel BuildModel(FolderClassificationConfig config)
        {
            var conv_arch = new[] { (1, 64), (1, 128), (2, 256), (2, 512), (2, 512) };

            var model = vgg(conv_arch, config.ClassNames.Length);

            // 如果需要用 model.summary(); 输出模型结构，需要先走一遍
            // 实际使用中不需要
            var tensor = tf.random.normal((1, config.InputShape[0], config.InputShape[1], 3));
            model.Apply(tensor);

            var optimizer = keras.optimizers.SGD();
            // var optimizer = keras.optimizers.Adam();
            var loss = keras.losses.SparseCategoricalCrossentropy();
            
            model.compile(optimizer, loss, new[] { "accuracy" });

            return model;
        }
    }
}
