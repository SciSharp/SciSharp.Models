using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;
using System.Linq;
using System.Diagnostics;
using Tensorflow;
using Tensorflow.NumPy;

namespace SciSharp.Models.ImageClassification
{
    public partial class ImageClassificationTask 
    {
        Dictionary<string, Dictionary<string, string[]>> image_dataset;
        Tensor resized_image_tensor;
        Tensor bottleneck_tensor;
        string CHECKPOINT_NAME;
        string tfhub_module = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3";
        Tensor final_tensor;
        Tensor ground_truth_input;
        int how_many_training_steps = 100;
        int train_batch_size = 100;
        Operation train_step;
        Tensor bottleneck_input;
        Tensor cross_entropy;

        // The location where variable checkpoints will be stored.

        string input_tensor_name = "Placeholder";
        string final_tensor_name = "Score";
        float learning_rate = 0.01f;

        int eval_step_interval = 10;

        int test_batch_size = -1;
        int validation_batch_size = 100;
        int intermediate_store_frequency = 0;
        int class_count = 0;
        const int MAX_NUM_IMAGES_PER_CLASS = 134217727;

        public void Train()
        {
            LoadData();

            var sw = new Stopwatch();
            using var graph = isImportingGraph ? ImportGraph() : BuildGraph();
            using var sess = tf.Session(graph);
            
            // Initialize all weights: for the module to their pretrained values,
            // and for the newly added retraining layer to random initial values.
            var init = tf.global_variables_initializer();
            sess.run(init);

            var (jpeg_data_tensor, decoded_image_tensor) = add_jpeg_decoding();

            // We'll make sure we've calculated the 'bottleneck' image summaries and
            // cached them on disk.
            cache_bottlenecks(sess, image_dataset, image_dir,
                    bottleneck_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor,
                    bottleneck_tensor, tfhub_module);

            // Create the operations we need to evaluate the accuracy of our new layer.
            var (evaluation_step, _) = add_evaluation_step(final_tensor, ground_truth_input);

            // Merge all the summaries and write them out to the summaries_dir
            var merged = tf.summary.merge_all();
            var train_writer = tf.summary.FileWriter(summaries_dir + "/train", sess.graph);
            var validation_writer = tf.summary.FileWriter(summaries_dir + "/validation", sess.graph);

            // Create a train saver that is used to restore values into an eval graph
            // when exporting models.
            var train_saver = tf.train.Saver();
            train_saver.save(sess, CHECKPOINT_NAME);

            sw.Restart();

            for (int i = 0; i < how_many_training_steps; i++)
            {
                var (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                        sess, image_dataset, train_batch_size, "training",
                        bottleneck_dir, image_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        tfhub_module);

                // Feed the bottlenecks and ground truth into the graph, and run a training
                // step. Capture training summaries for TensorBoard with the `merged` op.
                var results = sess.run(
                        new ITensorOrOperation[] { merged, train_step },
                        new FeedItem(bottleneck_input, train_bottlenecks),
                        new FeedItem(ground_truth_input, train_ground_truth));
                var train_summary = results[0];

                // TODO
                // train_writer.add_summary(train_summary, i);

                // Every so often, print out how well the graph is training.
                bool is_last_step = (i + 1 == how_many_training_steps);
                if ((i % eval_step_interval) == 0 || is_last_step)
                {
                    (float train_accuracy, float cross_entropy_value) = sess.run((evaluation_step, cross_entropy),
                        (bottleneck_input, train_bottlenecks),
                        (ground_truth_input, train_ground_truth));
                    print($"{DateTime.Now}: Step {i + 1}: Train accuracy = {train_accuracy * 100}%,  Cross entropy = {cross_entropy_value.ToString("G4")}");

                    var (validation_bottlenecks, validation_ground_truth, _) = get_random_cached_bottlenecks(
                        sess, image_dataset, validation_batch_size, "validation",
                        bottleneck_dir, image_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        tfhub_module);

                    // Run a validation step and capture training summaries for TensorBoard
                    // with the `merged` op.
                    (_, float validation_accuracy) = sess.run((merged, evaluation_step),
                        (bottleneck_input, validation_bottlenecks),
                        (ground_truth_input, validation_ground_truth));

                    // validation_writer.add_summary(validation_summary, i);
                    print($"{DateTime.Now}: Step {i + 1}: Validation accuracy = {validation_accuracy * 100}% (N={len(validation_bottlenecks)}) {sw.ElapsedMilliseconds}ms");
                    sw.Restart();
                }

                // Store intermediate results
                int intermediate_frequency = intermediate_store_frequency;
                if (intermediate_frequency > 0 && i % intermediate_frequency == 0 && i > 0)
                {

                }
            }

            // After training is complete, force one last save of the train checkpoint.
            train_saver.save(sess, CHECKPOINT_NAME);

            // We've completed all our training, so run a final test evaluation on
            // some new images we haven't used before.
            var (test_accuracy, predictions) = run_final_eval(sess, null, class_count, image_dataset,
                            jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                            bottleneck_tensor);

            // Write out the trained graph and labels with the weights stored as
            // constants.
            print($"Save final result to : {output_graph}");
            save_graph_to_file(output_graph, class_count);
            File.WriteAllText(output_label_path, string.Join("\n", image_dataset.Keys));
        }

        private (Tensor, Tensor) add_jpeg_decoding()
        {
            // height, width, depth
            var input_dim = (299, 299, 3);
            var jpeg_data = tf.placeholder(tf.@string, name: "DecodeJPGInput");
            var decoded_image = tf.image.decode_jpeg(jpeg_data, channels: input_dim.Item3);
            // Convert from full range of uint8 to range [0,1] of float32.
            var decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32);
            var decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0);
            var resize_shape = tf.stack(new int[] { input_dim.Item1, input_dim.Item2 });
            var resize_shape_as_int = tf.cast(resize_shape, dtype: tf.int32);
            var resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int);
            return (jpeg_data, resized_image);
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

        (NDArray, long[], string[]) get_random_cached_bottlenecks(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists,
            int how_many, string category, string bottleneck_dir, string image_dir,
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor,
            Tensor bottleneck_tensor, string module_name)
        {
            float[,] bottlenecks;
            var ground_truths = new List<long>();
            var filenames = new List<string>();
            class_count = image_lists.Keys.Count;
            if (how_many >= 0)
            {
                bottlenecks = new float[how_many, 2048];
                // Retrieve a random sample of bottlenecks.
                foreach (var unused_i in range(how_many))
                {
                    int label_index = new Random().Next(class_count);
                    string label_name = image_lists.Keys.ToArray()[label_index];
                    int image_index = new Random().Next(MAX_NUM_IMAGES_PER_CLASS);
                    string image_name = get_image_path(image_lists, label_name, image_index,
                                      image_dir, category);
                    var bottleneck = get_or_create_bottleneck(
                      sess, image_lists, label_name, image_index, image_dir, category,
                      bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name);
                    for (int col = 0; col < bottleneck.Length; col++)
                        bottlenecks[unused_i, col] = bottleneck[col];
                    ground_truths.Add(label_index);
                    filenames.Add(image_name);
                }
            }
            else
            {
                how_many = 0;
                // Retrieve all bottlenecks.
                foreach (var (label_index, label_name) in enumerate(image_lists.Keys.ToArray()))
                    how_many += image_lists[label_name][category].Length;
                bottlenecks = new float[how_many, 2048];

                var row = 0;
                foreach (var (label_index, label_name) in enumerate(image_lists.Keys.ToArray()))
                {
                    foreach (var (image_index, image_name) in enumerate(image_lists[label_name][category]))
                    {
                        var bottleneck = get_or_create_bottleneck(
                            sess, image_lists, label_name, image_index, image_dir, category,
                            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                            resized_input_tensor, bottleneck_tensor, module_name);

                        for (int col = 0; col < bottleneck.Length; col++)
                            bottlenecks[row, col] = bottleneck[col];
                        row++;
                        ground_truths.Add(label_index);
                        filenames.Add(image_name);
                    }
                }
            }

            return (bottlenecks, ground_truths.ToArray(), filenames.ToArray());
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
        private (float, NDArray) run_final_eval(Session train_session, object module_spec, int class_count,
            Dictionary<string, Dictionary<string, string[]>> image_lists,
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor,
            Tensor resized_image_tensor, Tensor bottleneck_tensor)
        {
            var (test_bottlenecks, test_ground_truth, test_filenames) = get_random_cached_bottlenecks(train_session, image_lists,
                                    test_batch_size, "testing", bottleneck_dir, image_dir, jpeg_data_tensor,
                                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor, tfhub_module);

            var (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step,
                prediction) = build_eval_session(class_count);

            (float accuracy, NDArray prediction1) = eval_session.run((evaluation_step, prediction),
                  (bottleneck_input, test_bottlenecks),
                  (ground_truth_input, test_ground_truth));

            print($"final test accuracy: {(accuracy * 100).ToString("G4")}% (N={len(test_bottlenecks)})");

            return (accuracy, prediction1);
        }

        private (Session, Tensor, Tensor, Tensor, Tensor, Tensor)
            build_eval_session(int class_count)
        {
            // If quantized, we need to create the correct eval graph for exporting.
            var graph = tf.Graph().as_default();
            var (bottleneck_tensor, resized_input_tensor, wants_quantization) = create_module_graph(graph);
            var eval_sess = tf.Session(graph);
            // Add the new layer for exporting.
            var (_, _, bottleneck_input, ground_truth_input, final_tensor) =
                add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                    wants_quantization, is_training: false);

            // Now we need to restore the values from the training graph to the eval
            // graph.
            tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME);

            var (evaluation_step, prediction) = add_evaluation_step(final_tensor,
                                                    ground_truth_input);

            return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input,
                evaluation_step, prediction);
        }

        private (Tensor, Tensor, bool) create_module_graph(Graph graph)
        {
            tf.train.import_meta_graph("graph/InceptionV3.meta");
            Tensor resized_input_tensor = graph.OperationByName(input_tensor_name); 
            Tensor bottleneck_tensor = graph.OperationByName("module_apply_default/hub_output/feature_vector/SpatialSqueeze");
            var wants_quantization = false;
            return (bottleneck_tensor, resized_input_tensor, wants_quantization);
        }
    }
}
