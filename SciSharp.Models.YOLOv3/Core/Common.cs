using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Layers;
using static Tensorflow.Binding;

namespace SciSharp.Models.YOLOv3
{
    class Common
    {
        public static Tensor convolutional(Tensor input_layer, TensorShape filters_shape,
            bool downsample = false, bool activate = true,
            bool bn = true)
        {
            int strides;
            string padding;

            if (downsample)
            {
                var zero_padding_2d = tf.keras.layers.ZeroPadding2D(new[,] { { 1, 0 }, { 1, 0 } });
                input_layer = zero_padding_2d.Apply(input_layer);
                strides = 2;
                padding = "valid";
            }
            else
            {
                strides = 1;
                padding = "same";
            }

            var conv2d_layer = tf.keras.layers.Conv2D(filters_shape[-1],
                kernel_size: filters_shape[0],
                strides: strides,
                padding: padding,
                use_bias: !bn,
                kernel_regularizer: tf.keras.regularizers.l2(0.0005f),
                kernel_initializer: tf.random_normal_initializer(stddev: 0.01f),
                bias_initializer: tf.constant_initializer(0f));
            var conv = conv2d_layer.Apply(input_layer);
            if (bn)
            {
                var batch_layer = new BatchNormalization(new BatchNormalizationArgs
                {
                });
                conv = batch_layer.Apply(conv);
            }

            if (activate)
                conv = tf.nn.leaky_relu(conv, alpha: 0.1f);

            return conv;
        }

        public static Tensor upsample(Tensor input_layer)
        {
            return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method: "nearest");
        }

        public static Tensor residual_block(Tensor input_layer, 
            int input_channel, int filter_num1, int filter_num2)
        {
            var short_cut = input_layer;

            var conv = convolutional(input_layer, (1, 1, input_channel, filter_num1));
            conv = convolutional(conv, (3, 3, filter_num1, filter_num2));

            var residual_output = short_cut + conv;

            return residual_output;
        }
    }
}
