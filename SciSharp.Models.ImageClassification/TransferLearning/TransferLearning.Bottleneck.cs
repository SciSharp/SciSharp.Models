using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;

namespace SciSharp.Models.ImageClassification
{
    public partial class TransferLearning 
    {
        /// <summary>
        /// Ensures all the training, testing, and validation bottlenecks are cached.
        /// </summary>
        /// <param name="sess"></param>
        /// <param name="image_lists"></param>
        /// <param name="bottleneck_dir"></param>
        /// <param name="jpeg_data_tensor"></param>
        /// <param name="decoded_image_tensor"></param>
        /// <param name="resized_image_tensor"></param>
        /// <param name="bottleneck_tensor"></param>
        /// <param name="tfhub_module"></param>
        void cache_bottlenecks(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists,
            string bottleneck_dir, Tensor jpeg_data_tensor, Tensor decoded_image_tensor,
            Tensor resized_input_tensor, Tensor bottleneck_tensor, string module_name)
        {
            int how_many_bottlenecks = 0;
            var kvs = image_lists.ToArray();
            var categories = new string[] { "training", "testing", "validation" };
            for(var i = 0; i < kvs.Length; i++)
            {
                var (label_name, label_lists) = (kvs[i].Key, kvs[i].Value);
                var sub_dir_path = Path.Combine(bottleneck_dir, label_name);
                Directory.CreateDirectory(sub_dir_path);

                for(var j = 0; j < categories.Length; j++)
                {
                    var category = categories[j];
                    var category_list = label_lists[category];
                    foreach (var (index, unused_base_name) in enumerate(category_list))
                    {
                        get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                            resized_input_tensor, bottleneck_tensor, module_name);
                        how_many_bottlenecks++;
                        if (how_many_bottlenecks % 300 == 0)
                            print($"{how_many_bottlenecks} bottleneck files created.");
                    }
                };
            };
        }

        float[] get_or_create_bottleneck(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists,
            string label_name, int index, string category, string bottleneck_dir,
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor,
            Tensor bottleneck_tensor, string module_name)
        {
            var label_lists = image_lists[label_name];
            string bottleneck_path = get_bottleneck_path(image_lists, label_name, bottleneck_dir, index, category, module_name);
            if (!File.Exists(bottleneck_path))
                return create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                                       category, sess, jpeg_data_tensor,
                                       decoded_image_tensor, resized_input_tensor,
                                       bottleneck_tensor);
            var bottleneck_string = File.ReadAllText(bottleneck_path);
            var bottleneck_values = Array.ConvertAll(bottleneck_string.Split(' '), x => float.Parse(x));
            return bottleneck_values;
        }

        float[] create_bottleneck_file(string bottleneck_path, Dictionary<string, Dictionary<string, string[]>> image_lists,
            string label_name, int index, string category, Session sess,
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor, Tensor bottleneck_tensor)
        {
            // Create a single bottleneck file.
            print("Creating bottleneck at " + bottleneck_path);
            var image_path = get_image_path(image_lists, label_name, _options.DataDir, index, category);
            if (!File.Exists(image_path))
                print($"File does not exist {image_path}");

            var image_data = File.ReadAllBytes(image_path);
            var bottleneck_values = run_bottleneck_on_image(
                sess, image_data, jpeg_data_tensor, decoded_image_tensor,
                resized_input_tensor, bottleneck_tensor);
            var values = bottleneck_values.ToArray<float>();
            var bottleneck_string = string.Join(" ", values);
            File.WriteAllText(bottleneck_path, bottleneck_string);
            return values;
        }

        /// <summary>
        /// Runs inference on an image to extract the 'bottleneck' summary layer.
        /// </summary>
        /// <param name="sess">Current active TensorFlow Session.</param>
        /// <param name="image_data">Data of raw JPEG data.</param>
        /// <param name="image_data_tensor">Input data layer in the graph.</param>
        /// <param name="decoded_image_tensor">Output of initial image resizing and preprocessing.</param>
        /// <param name="resized_input_tensor">The input node of the recognition graph.</param>
        /// <param name="bottleneck_tensor">Layer before the final softmax.</param>
        /// <returns></returns>
        NDArray run_bottleneck_on_image(Session sess, byte[] image_data, Tensor image_data_tensor,
                            Tensor decoded_image_tensor, Tensor resized_input_tensor, Tensor bottleneck_tensor)
        {
            // First decode the JPEG image, resize it, and rescale the pixel values.
            var resized_input_values = sess.run(decoded_image_tensor, new FeedItem(image_data_tensor, new NDArray(image_data, image_data_tensor.shape, dtype: tf.@string)));
            // Then run it through the recognition network.
            var bottleneck_values = sess.run(bottleneck_tensor, new FeedItem(resized_input_tensor, resized_input_values))[0];
            bottleneck_values = np.squeeze(bottleneck_values);
            return bottleneck_values;
        }

        string get_image_path(Dictionary<string, Dictionary<string, string[]>> image_lists, string label_name,
            string image_dir, int index, string category)
        {
            if (!image_lists.ContainsKey(label_name))
                print($"Label does not exist {label_name}");

            var label_lists = image_lists[label_name];
            if (!label_lists.ContainsKey(category))
                print($"Category does not exist {category}");
            var category_list = label_lists[category];
            if (category_list.Length == 0)
                print($"Label {label_name} has no images in the category {category}.");

            var mod_index = index % len(category_list);
            var base_name = category_list[mod_index].Split(Path.DirectorySeparatorChar).Last();
            var sub_dir = label_name;
            var full_path = Path.Combine(image_dir, sub_dir, base_name);
            return full_path;
        }

        string get_bottleneck_path(Dictionary<string, Dictionary<string, string[]>> image_lists, string label_name,
            string image_dir, int index, string category, string module_name)
        {
            module_name = (module_name.Replace("://", "~")  // URL scheme.
                 .Replace('/', '~')  // URL and Unix paths.
                 .Replace(':', '~').Replace('\\', '~'));  // Windows paths.
            return get_image_path(image_lists, label_name, image_dir, index, category) + "_" + module_name + ".txt";
        }
        (NDArray, long[], string[]) get_random_cached_bottlenecks(Session sess, Dictionary<string, Dictionary<string, string[]>> image_lists,
            int how_many, string category, string bottleneck_dir,
            Tensor jpeg_data_tensor, Tensor decoded_image_tensor, Tensor resized_input_tensor,
            Tensor bottleneck_tensor, string module_name)
        {
            float[,] bottlenecks;
            var ground_truths = new List<long>();
            var filenames = new List<string>();
            var class_count = image_lists.Keys.Count;
            if (how_many >= 0)
            {
                bottlenecks = new float[how_many, 2048];
                // Retrieve a random sample of bottlenecks.
                foreach (var unused_i in range(how_many))
                {
                    int label_index = new Random().Next(class_count);
                    string label_name = image_lists.Keys.ToArray()[label_index];
                    int image_index = new Random().Next(MAX_NUM_IMAGES_PER_CLASS);
                    string image_name = get_image_path(image_lists, label_name, bottleneck_dir, image_index, category);
                    var bottleneck = get_or_create_bottleneck(
                      sess, image_lists, label_name, image_index, category,
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
                            sess, image_lists, label_name, image_index, category,
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
    }
}
