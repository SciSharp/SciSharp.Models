using Newtonsoft.Json;
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
    public class GoogLeNet : IModelZoo
    {
        public class Inception : Layer
        {
            static int layerId = 0;

            [JsonObject(MemberSerialization.OptIn)]
            public class InceptionLayerArgs : LayerArgs
            {
                public int c1;

                public int[] c2 = new int[2];

                public int[] c3 = new int[2];

                public int c4;
            }

            ILayer p1_1;

            ILayer p2_1;

            ILayer p2_2;

            ILayer p3_1;

            ILayer p3_2;

            ILayer p4_1;

            ILayer p4_2;

            public Inception(InceptionLayerArgs args) : base(args)
            {
                if (string.IsNullOrEmpty(args.Name))
                {
                    args.Name = "Inception_" + ++layerId;
                }

                p1_1 = keras.layers.Conv2D(args.c1, 1, activation: "relu");

                p2_1 = keras.layers.Conv2D(args.c2[0], 1, activation: "relu");
                p2_2 = keras.layers.Conv2D(args.c2[1], 3, activation: "relu", padding: "same");

                p3_1 = keras.layers.Conv2D(args.c3[0], 1, activation: "relu");
                p3_2 = keras.layers.Conv2D(args.c3[1], 5, activation: "relu", padding: "same");

                p4_1 = keras.layers.MaxPooling2D(3, 1, padding: "same");
                p4_2 = keras.layers.Conv2D(args.c4, 1, activation: "relu");

                Layers.add(p1_1);
                Layers.add(p2_1);
                Layers.add(p2_2);
                Layers.add(p3_1);
                Layers.add(p3_2);
                Layers.add(p4_1);
                Layers.add(p4_2);
            }

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var p1 = p1_1.Apply(inputs);
                var p2 = p2_2.Apply(p2_1.Apply(inputs));
                var p3 = p3_2.Apply(p3_1.Apply(inputs));
                var p4 = p4_2.Apply(p4_1.Apply(inputs));

                var x = new Tensors(p1, p2, p3, p4);

                return keras.layers.Concatenate().Apply(x, state, training, optional_args);
            }


        }

        public  IModel BuildModel(FolderClassificationConfig config)
        {
            var b1 = () =>
            {
                var ret = new BlocksLayer(new[]{
                    keras.layers.Conv2D(64, 7, strides: 2, padding: "same", activation: "relu"),
                    keras.layers.MaxPooling2D(pool_size: 3, strides: 2, padding:"same")
                });
                return ret;
            };


            var b2 = () => {
                var ret = new BlocksLayer(new[] {
                    keras.layers.Conv2D(filters: 64, 1, activation: "relu"),
                    keras.layers.Conv2D(filters: 192, 3, padding: "same", activation: "relu"),
                    keras.layers.MaxPooling2D(pool_size: 3, strides: 2, padding: "same")
                });

                return ret;
            };

            var b3 = () => {
                var ret = new BlocksLayer(new[] {
                    new Inception(new Inception.InceptionLayerArgs { c1 = 64, c2 = new[]{ 96, 128 }, c3 = new[]{ 16, 32 }, c4 = 32 }),
                    new Inception(new Inception.InceptionLayerArgs { c1 = 128, c2 = new[]{ 128, 192 }, c3 = new[]{ 32, 96 }, c4 = 64 }),
                    keras.layers.MaxPooling2D(pool_size: 3, strides: 2, padding: "same")
                });

                return ret;
            };

            var b4 = () => {
                var ret = new BlocksLayer(new[] {
                  new Inception(new Inception.InceptionLayerArgs { c1 = 192, c2 = new[]{ 96, 208 }, c3 = new[]{ 16, 48 }, c4 = 64 }),
                  new Inception(new Inception.InceptionLayerArgs { c1 = 160, c2 = new[]{ 112, 224 }, c3 = new[]{ 24, 64 }, c4 = 64 }),
                  new Inception(new Inception.InceptionLayerArgs { c1 = 128, c2 = new[]{ 128, 256 }, c3 = new[]{ 24, 64 }, c4 = 64 }),
                  new Inception(new Inception.InceptionLayerArgs { c1 = 112, c2 = new[]{ 144, 288 }, c3 = new[]{ 32, 64 }, c4 = 64 }),
                  new Inception(new Inception.InceptionLayerArgs { c1 = 256, c2 = new[]{ 160, 320 }, c3 = new[]{ 32, 128 }, c4 = 128 }),
                  keras.layers.MaxPooling2D(pool_size: 3, strides: 2, padding: "same")
                });

                return ret;
            };

            var b5 = () => {
                var ret = new BlocksLayer(new[]
                {
                  new Inception(new Inception.InceptionLayerArgs { c1 = 256, c2 = new[]{ 160, 320 }, c3 = new[]{ 32, 128 }, c4 = 128 }),
                  new Inception(new Inception.InceptionLayerArgs { c1 = 384, c2 = new[]{ 192, 384 }, c3 = new[]{ 48, 128 }, c4 = 128 }),
                  keras.layers.GlobalAveragePooling2D(),
                  keras.layers.Flatten()
                }); ;
                return ret;
            };

            var model = keras.Sequential(new[] {
                b1(), b2(), b3(), b4(), b5(),
                keras.layers.Dense(config.NumberOfClass)});

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
