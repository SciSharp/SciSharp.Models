using Tensorflow;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.ImageClassification.Zoo
{
    public class MobilenetV2 : IModelZoo
    {
        static int _id = 1;

        Tensor inverted_residual_block(Tensor input, int filters, int strides, int expansion)
        {
            var x = input;

            if (strides == 1)
            {
                x = keras.layers.Conv2D(filters * expansion * 2, kernel_size: 1, strides: 1, padding: "same", activation: "relu6").Apply(x);
            }

            // Depthwise convolution phase
            x = keras.layers.DepthwiseConv2D(kernel_size: 3, strides: strides, padding: "same").Apply(x);
            x = keras.layers.BatchNormalization(name:$"bn_{filters}_{strides}_{expansion}").Apply(x);
            x = keras.layers.ReLU6().Apply(x);

            // Project phase
            x = keras.layers.Conv2D(filters * expansion * 2, kernel_size: 1, strides:1, padding: "same").Apply(x);

            // Skip connection
            if (strides == 1)
            {
                x = keras.layers.Add().Apply(new[] { x, input });
            }

            return x;
        }

        public IModel BuildModel(FolderClassificationConfig config)
        {
            Shape input_shape = (config.InputShape[0], config.InputShape[1], 3);
            int num_classes = config.NumberOfClass;

            // Define the input tensor
            var inputs = keras.Input(input_shape);

            // Initial convolution block
            var x = keras.layers.Conv2D(32, kernel_size: (3, 3), strides: (2, 2), padding: "same", activation:"relu6").Apply(inputs);

            // Inverted residual blocks
            x = inverted_residual_block(x, filters: 16, strides: 1, expansion: 1);

            x = inverted_residual_block(x, filters: 24, strides: 2, expansion: 6);
            x = inverted_residual_block(x, filters: 24, strides: 1, expansion: 6);

            x = inverted_residual_block(x, filters: 32, strides: 2, expansion: 6);
            x = inverted_residual_block(x, filters: 32, strides: 1, expansion: 6);
            x = inverted_residual_block(x, filters: 32, strides: 1, expansion: 6);

            x = inverted_residual_block(x, filters: 64, strides: 2, expansion: 6);
            x = inverted_residual_block(x, filters: 64, strides: 1, expansion: 6);
            x = inverted_residual_block(x, filters: 64, strides: 1, expansion: 6);
            x = inverted_residual_block(x, filters: 64, strides: 1, expansion: 6);

            x = inverted_residual_block(x, filters: 96, strides: 2, expansion: 6);
            x = inverted_residual_block(x, filters: 96, strides: 1, expansion: 6);
            x = inverted_residual_block(x, filters: 96, strides: 1, expansion: 6);

            //x = inverted_residual_block(x, filters: 160, strides: 2, expansion: 6);
            //x = inverted_residual_block(x, filters: 160, strides: 1, expansion: 6);
            //x = inverted_residual_block(x, filters: 160, strides: 1, expansion: 6);
            //x = inverted_residual_block(x, filters: 320, strides: 1, expansion: 6);

            // Define the output tensor
            x = keras.layers.Conv2D(1280, kernel_size: (1, 1), padding: "same", activation:"relu").Apply(x);
            x = keras.layers.GlobalAveragePooling2D().Apply(x);
            var outputs = keras.layers.Dense(num_classes, activation: "softmax").Apply(x);

            // Create the model
            var model = keras.Model(inputs, outputs, "MobilenetV2");
            model.build(input_shape: input_shape);

            var optimizer = keras.optimizers.Adam();
            var loss = keras.losses.SparseCategoricalCrossentropy(from_logits: true);

            model.compile(optimizer, loss, new[] { "accuracy" });

            return model;
        }
    }
}
