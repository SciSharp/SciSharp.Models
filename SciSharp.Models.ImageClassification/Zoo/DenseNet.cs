using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.ImageClassification.Zoo
{
    public class DenseNet : IModelZoo
    {
        class ConvBlock : BlocksLayer
        {
            static int layerId;

            public ConvBlock(int num_channels) : base(new LayerArgs { Name = "ConvBlock" + ++layerId })
            {
                Layers.add(keras.layers.BatchNormalization());
                Layers.add(keras.layers.LeakyReLU());
                Layers.add(keras.layers.Conv2D(filters: num_channels, kernel_size: (3, 3), padding: "same", activation: "relu"));
            }

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                // print($"layer.name {this.Name} x.shape:{inputs.shape}");
                var y = inputs;

                foreach (var lay in Layers)
                {
                    y = lay.Apply(y, state, training, optional_args);
                }

                return keras.layers.Concatenate().Apply((inputs, y), state, training, optional_args);
            }
        }

        public class DenseBlock : BlocksLayer
        {
            static int layerId;

            public DenseBlock(int num_convs, int num_channels) : base(new LayerArgs { Name = "DenseBlock" + ++layerId })
            {
                for (var i = 0; i < num_convs; i++)
                {
                    Layers.add(new ConvBlock(num_channels));
                }
            }

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var x = inputs;
                foreach (var layer in Layers)
                {
                    x = layer.Apply(x, state, training, optional_args);
                }

                return x;
            }
        }
        class TransitionBlock : BlocksLayer
        {
            static int layerId;

            public TransitionBlock(int num_channels) : base(new LayerArgs { Name = "TransitionBlock" + ++layerId })
            {
                Layers.add(keras.layers.BatchNormalization());
                Layers.add(keras.layers.LeakyReLU());
                Layers.add(keras.layers.Conv2D(num_channels, kernel_size: 1, activation: "relu"));
                Layers.add(keras.layers.AveragePooling2D(pool_size: 2, strides: 2));
            }

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var x = inputs;
                foreach (var layer in Layers)
                {
                    x = layer.Apply(x, state, training, optional_args);
                }
                return x;
            }
        }

        public IModel BuildModel(FolderClassificationConfig config)
        {
            var blocks = () => {
                var layers = new List<ILayer>();
                layers.AddRange(new[] {
                                keras.layers.Conv2D(64, kernel_size: 7, strides: 2, padding: "same", activation: "relu"),
                                keras.layers.BatchNormalization(),
                                keras.layers.LeakyReLU(),
                                keras.layers.MaxPooling2D(pool_size: 3, strides: 2, padding: "same")
                                });

                var num_channels = 64;
                var growth_rate = 32;
                var num_convs_in_dense_blocks = new[] { 4, 4, 4, 4 };

                for (var i = 0; i < num_convs_in_dense_blocks.Length; i++)
                {
                    var num_convs = num_convs_in_dense_blocks[i];

                    layers.add(new DenseBlock(num_convs, growth_rate));
                    num_channels += (num_convs * growth_rate);
                    if (i != num_convs_in_dense_blocks.Length - 1)
                    {
                        num_channels = num_channels / 2;
                        layers.add(new TransitionBlock(num_channels));
                    }
                }

                return new BlocksLayer(layers);
            };

            var model = keras.Sequential(new[] {
                blocks(),
                keras.layers.BatchNormalization(),
                keras.layers.LeakyReLU(),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Flatten(),
                keras.layers.Dense(config.NumberOfClass),
            });

            var X = tf.zeros((1, config.InputShape[0], config.InputShape[1], 3));
            model.Apply(X); // 需要走一遍

            // var optimizer = keras.optimizers.SGD();
            var optimizer = keras.optimizers.Adam();
            var loss = keras.losses.SparseCategoricalCrossentropy(from_logits: true);
            model.compile(optimizer, loss, new[] { "accuracy" });


            return model;
        }
    }
}
