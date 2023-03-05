using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Keras.Utils;
using System.Linq;
using System.Diagnostics;
using Tensorflow;
using Tensorflow.NumPy;
using SciSharp.Models.Exceptions;

namespace SciSharp.Models.ImageClassification
{
    public partial class TransferLearning 
    {
        public ModelTestResult Test(TestingOptions options)
        {
            if (!File.Exists(_options.ModelPath))
                throw new FreezedGraphNotFoundException();

            image_dataset = LoadDataFromDir(_options.DataDir, testingPercentage: 1.0f, validationPercentage: 0);

            var graph = tf.Graph().as_default();
            graph.Import(_options.ModelPath);

            resized_image_tensor = graph.OperationByName(input_tensor_name);
            bottleneck_tensor = graph.OperationByName("module_apply_default/hub_output/feature_vector/SpatialSqueeze");
            wants_quantization = false;

            var (jpeg_data_tensor, decoded_image_tensor) = add_jpeg_decoding();

            var sess = tf.Session(graph);

            var (test_accuracy, predictions) = run_final_eval(sess, image_dataset,
                           jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                           bottleneck_tensor);

            return new ModelTestResult
            {
                Accuracy = test_accuracy,
                Predictions = predictions
            };
        }

        /// <summary>
        /// Runs a final evaluation on an eval graph using the test data set.
        /// </summary>
        /// <param name="train_session"></param>
        /// <param name="module_spec"></param>
        /// <param name="class_count"></param>
        /// <param name="image_lists"></param>
        /// <param name="jpeg_data_tensor"></param>
        /// <param name="decoded_image_tensor"></param>
        /// <param name="resized_image_tensor"></param>
        /// <param name="bottleneck_tensor"></param>
        (float, NDArray) run_final_eval(Session train_session,
            Dictionary<string, Dictionary<string, string[]>> image_lists,
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor,
            Tensor resized_image_tensor, Tensor bottleneck_tensor)
        {
            var (test_bottlenecks, test_ground_truth, test_filenames) = get_random_cached_bottlenecks(train_session, image_lists,
                                    test_batch_size, "testing", bottleneck_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor, tfhub_module);

            var (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step,
                prediction) = build_eval_session();

            (float accuracy, NDArray prediction1) = eval_session.run((evaluation_step, prediction),
                  (bottleneck_input, test_bottlenecks),
                  (ground_truth_input, test_ground_truth));

            print($"final test accuracy: {(accuracy * 100).ToString("G4")}% (N={len(test_bottlenecks)})");

            return (accuracy, prediction1);
        }

        (Session, Tensor, Tensor, Tensor, Tensor, Tensor) build_eval_session()
        {
            // If quantized, we need to create the correct eval graph for exporting.
            var graph = tf.Graph().as_default();
            var (bottleneck_tensor, resized_input_tensor, wants_quantization) = create_module_graph(graph);
            var eval_sess = tf.Session(graph);
            // Add the new layer for exporting.
            var (_, _, bottleneck_input, ground_truth_input, final_tensor) =
                add_final_retrain_ops(len(image_dataset), final_tensor_name, bottleneck_tensor,
                    wants_quantization, is_training: false);

            // Now we need to restore the values from the training graph to the eval
            // graph.
            tf.train.Saver().restore(eval_sess, Path.Combine(taskDir, "checkpoint"));

            var (evaluation_step, prediction) = add_evaluation_step(final_tensor,
                                                    ground_truth_input);

            return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
                evaluation_step, prediction);
        }

        /// <summary>
        /// Inserts the operations we need to evaluate the accuracy of our results.
        /// </summary>
        /// <param name="result_tensor"></param>
        /// <param name="ground_truth_tensor"></param>
        /// <returns></returns>
        (Tensor, Tensor) add_evaluation_step(Tensor result_tensor, Tensor ground_truth_tensor)
        {
            Tensor evaluation_step = null, correct_prediction = null, prediction = null;

            tf_with(tf.name_scope("accuracy"), scope =>
            {
                tf_with(tf.name_scope("correct_prediction"), delegate
                {
                    prediction = tf.math.argmax(result_tensor, 1);
                    correct_prediction = tf.equal(prediction, ground_truth_tensor);
                });

                tf_with(tf.name_scope("accuracy"), delegate
                {
                    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
                });
            });

            tf.summary.scalar("accuracy", evaluation_step);
            return (evaluation_step, prediction);
        }
    }
}
