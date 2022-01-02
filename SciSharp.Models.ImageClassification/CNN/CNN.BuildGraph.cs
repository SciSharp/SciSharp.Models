using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace SciSharp.Models.ImageClassification
{
    public partial class CNN
    {
        public GraphBuiltResult BuildGraph(TrainingOptions options)
        {
            var result = new GraphBuiltResult
            {
                Graph = new Graph().as_default()
            };

            tf_with(tf.name_scope("Input"), delegate
            {
                // Placeholders for features (x) and lables(y)
                var dims = new List<long>(_options.InputShape.dims);
                dims.Insert(0, -1);
                result.Features = tf.placeholder(tf.float32, shape: new Shape(dims.ToArray()), name: "X");
                result.Labels = tf.placeholder(tf.float32, shape: (-1, _options.NumberOfClass), name: "Y");
            });

            var conv1 = conv_layer(result.Features, _convArgs.FilterSize1, _convArgs.NumberOfFilters1, _convArgs.Stride1, name: "conv1");
            var pool1 = max_pool(conv1, ksize: 2, stride: 2, name: "pool1");
            var conv2 = conv_layer(pool1, _convArgs.FilterSize2, _convArgs.NumberOfFilters2, _convArgs.Stride2, name: "conv2");
            var pool2 = max_pool(conv2, ksize: 2, stride: 2, name: "pool2");
            var layer_flat = flatten_layer(pool2);
            var fc1 = fc_layer(layer_flat, _convArgs.NumberOfNeurons, "FC1", use_relu: true);
            var output_logits = fc_layer(fc1, _options.NumberOfClass, "OUT", use_relu: false);

            tf_with(tf.variable_scope("Train"), delegate
            {
                result.Loss = tf_with(tf.variable_scope("Loss"), x
                    => tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels: result.Labels, 
                        logits: output_logits), name: "loss"));

                result.Optimizer = tf_with(tf.variable_scope("Optimizer"), x
                    => tf.train.AdamOptimizer(learning_rate: options.LearningRate, name: "Adam-op")
                        .minimize(result.Loss));

                result.Accuracy = tf_with(tf.variable_scope("Accuracy"), delegate
                {
                    var correct_prediction = tf.equal(tf.math.argmax(output_logits, 1), tf.math.argmax(result.Labels, 1), name: "correct_pred");
                    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name: "accuracy");
                });

                result.Prediction = tf_with(tf.variable_scope("Prediction"), x
                    => tf.math.argmax(output_logits, axis: 1, name: "predictions"));
            });

            return result;
        }

        /// <summary>
        /// Create a 2D convolution layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="filter_size">size of each filter</param>
        /// <param name="num_filters">number of filters(or output feature maps)</param>
        /// <param name="stride">filter stride</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor conv_layer(Tensor x, int filter_size, int num_filters, int stride, string name)
        {
            return tf_with(tf.variable_scope(name), delegate
            {
                var num_in_channel = x.shape[x.ndim - 1];
                var shape = new int[] { filter_size, filter_size, (int)num_in_channel, num_filters };
                var W = weight_variable("W", shape);
                // var tf.summary.histogram("weight", W);
                var b = bias_variable("b", new[] { num_filters });
                // tf.summary.histogram("bias", b);
                var layer = tf.nn.conv2d(x, W.AsTensor(),
                                     strides: new int[] { 1, stride, stride, 1 },
                                     padding: "SAME");
                layer += b.AsTensor();
                return tf.nn.relu(layer);
            });
        }

        /// <summary>
        /// Create a weight variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private IVariableV1 weight_variable(string name, int[] shape)
        {
            var initer = tf.truncated_normal_initializer(stddev: 0.01f);
            return tf.compat.v1.get_variable(name,
                                   dtype: tf.float32,
                                   shape: shape,
                                   initializer: initer);
        }

        /// <summary>
        /// Create a bias variable with appropriate initialization
        /// </summary>
        /// <param name="name"></param>
        /// <param name="shape"></param>
        /// <returns></returns>
        private IVariableV1 bias_variable(string name, int[] shape)
        {
            var initial = tf.constant(0f, shape: shape, dtype: tf.float32);
            return tf.compat.v1.get_variable(name,
                           dtype: tf.float32,
                           initializer: initial);
        }

        /// <summary>
        /// Create a max pooling layer
        /// </summary>
        /// <param name="x">input to max-pooling layer</param>
        /// <param name="ksize">size of the max-pooling filter</param>
        /// <param name="stride">stride of the max-pooling filter</param>
        /// <param name="name">layer name</param>
        /// <returns>The output array</returns>
        private Tensor max_pool(Tensor x, int ksize, int stride, string name)
        {
            return tf.nn.max_pool(x,
                ksize: new[] { 1, ksize, ksize, 1 },
                strides: new[] { 1, stride, stride, 1 },
                padding: "SAME",
                name: name);
        }

        /// <summary>
        /// Flattens the output of the convolutional layer to be fed into fully-connected layer
        /// </summary>
        /// <param name="layer">input array</param>
        /// <returns>flattened array</returns>
        private Tensor flatten_layer(Tensor layer)
        {
            return tf_with(tf.variable_scope("Flatten_layer"), delegate
            {
                var layer_shape = layer.shape;
                var num_features = layer_shape[new Slice(1, 4)].size;
                var layer_flat = tf.reshape(layer, new[] { -1, num_features });

                return layer_flat;
            });
        }

        /// <summary>
        /// Create a fully-connected layer
        /// </summary>
        /// <param name="x">input from previous layer</param>
        /// <param name="num_units">number of hidden units in the fully-connected layer</param>
        /// <param name="name">layer name</param>
        /// <param name="use_relu">boolean to add ReLU non-linearity (or not)</param>
        /// <returns>The output array</returns>
        private Tensor fc_layer(Tensor x, int num_units, string name, bool use_relu = true)
        {
            return tf_with(tf.variable_scope(name), delegate
            {
                var in_dim = x.shape[1];

                var W = weight_variable("W_" + name, shape: new[] { (int)in_dim, num_units });
                var b = bias_variable("b_" + name, new[] { num_units });

                var layer = tf.matmul(x, W.AsTensor()) + b.AsTensor();
                if (use_relu)
                    layer = tf.nn.relu(layer);

                return layer;
            });
        }
    }
}
