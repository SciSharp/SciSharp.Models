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
    public partial class TransferLearning 
    {
        Dictionary<string, Dictionary<string, string[]>> image_dataset;
        Tensor resized_image_tensor;
        Tensor bottleneck_tensor;
        string tfhub_module = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/3";
        Tensor final_tensor;
        Tensor ground_truth_input;

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
        const int MAX_NUM_IMAGES_PER_CLASS = 134217727;

        public void Train(TrainingOptions options)
        {
            image_dataset = LoadDataFromDir(this.options.DataDir, 
                testingPercentage: this.options.TestingPercentage,
                validationPercentage: this.options.ValidationPercentage);

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
            cache_bottlenecks(sess, image_dataset,
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
            var checkpoint = Path.Combine(taskDir, "checkpoint");
            train_saver.save(sess, checkpoint);

            sw.Restart();

            for (int i = 0; i < options.TrainingSteps; i++)
            {
                var (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
                        sess, image_dataset, options.BatchSize, "training",
                        bottleneck_dir, jpeg_data_tensor,
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
                bool is_last_step = (i + 1 == options.TrainingSteps);
                if ((i % eval_step_interval) == 0 || is_last_step)
                {
                    (float train_accuracy, float cross_entropy_value) = sess.run((evaluation_step, cross_entropy),
                        (bottleneck_input, train_bottlenecks),
                        (ground_truth_input, train_ground_truth));

                    var (validation_bottlenecks, validation_ground_truth, _) = get_random_cached_bottlenecks(
                        sess, image_dataset, validation_batch_size, "validation",
                        bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                        tfhub_module);

                    // Run a validation step and capture training summaries for TensorBoard
                    // with the `merged` op.
                    (_, float validation_accuracy) = sess.run((merged, evaluation_step),
                        (bottleneck_input, validation_bottlenecks),
                        (ground_truth_input, validation_ground_truth));

                    // validation_writer.add_summary(validation_summary, i);
                    print($"Step {i}: Training accuracy = {train_accuracy * 100}%, Cross entropy = {cross_entropy_value.ToString("G4")}, Validation accuracy = {validation_accuracy * 100}% (N={len(validation_bottlenecks)}) {sw.ElapsedMilliseconds}ms");
                    sw.Restart();
                }

                // Store intermediate results
                int intermediate_frequency = intermediate_store_frequency;
                if (intermediate_frequency > 0 && i % intermediate_frequency == 0 && i > 0)
                {

                }
            }

            // After training is complete, force one last save of the train checkpoint.
            print($"Saving checkpoint to {checkpoint}");
            train_saver.save(sess, checkpoint);

            SaveModel();
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

        (Tensor, Tensor, bool) create_module_graph(Graph graph)
        {
            tf.train.import_meta_graph("graph/InceptionV3.meta");
            Tensor resized_input_tensor = graph.OperationByName(input_tensor_name); 
            Tensor bottleneck_tensor = graph.OperationByName("module_apply_default/hub_output/feature_vector/SpatialSqueeze");
            var wants_quantization = false;
            return (bottleneck_tensor, resized_input_tensor, wants_quantization);
        }
    }
}
