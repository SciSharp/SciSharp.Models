using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow;

namespace SciSharp.Models.ImageClassification
{
    public partial class TransferLearning 
    {
        bool wants_quantization;

        Graph ImportGraph()
        {
            var graph = tf.Graph().as_default();
            // Set up the pre-trained graph.
            (bottleneck_tensor, resized_image_tensor, wants_quantization) = create_module_graph(graph);

            // Add the new layer that we'll be training.
            (train_step, cross_entropy, bottleneck_input,
             ground_truth_input, final_tensor) = add_final_retrain_ops(
                 len(image_dataset), final_tensor_name, bottleneck_tensor,
                 wants_quantization, is_training: true);

            return graph;
        }

        /// <summary>
        /// Adds a new softmax and fully-connected layer for training and eval.
        /// 
        /// We need to retrain the top layer to identify our new classes, so this function
        /// adds the right operations to the graph, along with some variables to hold the
        /// weights, and then sets up all the gradients for the backward pass.
        /// 
        /// The set up for the softmax and fully-connected layers is based on:
        /// https://www.tensorflow.org/tutorials/mnist/beginners/index.html
        /// </summary>
        /// <param name="class_count"></param>
        /// <param name="final_tensor_name"></param>
        /// <param name="bottleneck_tensor"></param>
        /// <param name="quantize_layer"></param>
        /// <param name="is_training"></param>
        /// <returns></returns>
        private (Operation, Tensor, Tensor, Tensor, Tensor) add_final_retrain_ops(int class_count, string final_tensor_name,
            Tensor bottleneck_tensor, bool quantize_layer, bool is_training)
        {
            var (batch_size, bottleneck_tensor_size) = (bottleneck_tensor.shape.dims[0], bottleneck_tensor.shape.dims[1]);
            tf_with(tf.name_scope("input"), scope =>
            {
                bottleneck_input = tf.placeholder_with_default(
                    bottleneck_tensor,
                    shape: bottleneck_tensor.shape,
                    name: "BottleneckInputPlaceholder");

                ground_truth_input = tf.placeholder(tf.int64, new Shape(batch_size), name: "GroundTruthInput");
            });

            // Organizing the following ops so they are easier to see in TensorBoard.
            string layer_name = "final_retrain_ops";
            Tensor logits = null;
            tf_with(tf.name_scope(layer_name), scope =>
            {
                IVariableV1 layer_weights = null;
                tf_with(tf.name_scope("weights"), delegate
                {
                    var initial_value = tf.truncated_normal((bottleneck_tensor_size, class_count), stddev: 0.001f);
                    layer_weights = tf.Variable(initial_value, name: "final_weights");
                    variable_summaries(layer_weights.AsTensor());
                });

                IVariableV1 layer_biases = null;
                tf_with(tf.name_scope("biases"), delegate
                {
                    layer_biases = tf.Variable(tf.zeros(new Shape(class_count)), name: "final_biases");
                    variable_summaries(layer_biases.AsTensor());
                });

                tf_with(tf.name_scope("Wx_plus_b"), delegate
                {
                    logits = tf.matmul(bottleneck_input, layer_weights.AsTensor()) + layer_biases.AsTensor();
                    tf.summary.histogram("pre_activations", logits);
                });
            });

            final_tensor = tf.nn.softmax(logits, name: final_tensor_name);

            // The tf.contrib.quantize functions rewrite the graph in place for
            // quantization. The imported model graph has already been rewritten, so upon
            // calling these rewrites, only the newly added final layer will be
            // transformed.
            if (quantize_layer)
            {
                throw new NotImplementedException("quantize_layer");
                /*if (is_training)
                    tf.contrib.quantize.create_training_graph();
                else
                    tf.contrib.quantize.create_eval_graph();*/
            }

            tf.summary.histogram("activations", final_tensor);

            // If this is an eval graph, we don't need to add loss ops or an optimizer.
            if (!is_training)
                return (null, null, bottleneck_input, ground_truth_input, final_tensor);

            Tensor cross_entropy_mean = null;
            tf_with(tf.name_scope("cross_entropy"), delegate
            {
                cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
                    labels: ground_truth_input, logits: logits);
            });

            tf.summary.scalar("cross_entropy", cross_entropy_mean);

            tf_with(tf.name_scope("train"), delegate
            {
                var optimizer = tf.train.GradientDescentOptimizer(learning_rate);
                train_step = optimizer.minimize(cross_entropy_mean);
            });

            return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
                final_tensor);
        }

        void variable_summaries(Tensor var)
        {
            tf_with(tf.name_scope("summaries"), delegate
            {
                var mean = tf.reduce_mean(var);
                tf.summary.scalar("mean", mean);
                Tensor stddev = null;
                tf_with(tf.name_scope("stddev"), delegate
                {
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)));
                });
                tf.summary.scalar("stddev", stddev);
                tf.summary.scalar("max", tf.reduce_max(var));
                tf.summary.scalar("min", tf.reduce_min(var));
                tf.summary.histogram("histogram", var);
            });
        }

        Graph BuildGraph() => throw new NotImplementedException("");
    }
}
