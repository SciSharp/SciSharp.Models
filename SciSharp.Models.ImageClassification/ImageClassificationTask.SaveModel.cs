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
using Google.Protobuf;

namespace SciSharp.Models.ImageClassification
{
    public partial class ImageClassificationTask 
    {
        public void SaveModel()
        {
            // Write out the trained graph and labels with the weights stored as
            // constants.
            print($"Saving final result to: {output_graph}");
            save_graph_to_file(output_graph, class_count);
            File.WriteAllText(output_label_path, string.Join("\n", image_dataset.Keys));
        }

        /// <summary>
        /// Saves an graph to file, creating a valid quantized one if necessary.
        /// </summary>
        /// <param name="graph_file_name"></param>
        /// <param name="class_count"></param>
        void save_graph_to_file(string graph_file_name, int class_count)
        {
            var (sess, _, _, _, _, _) = build_eval_session(class_count);
            var graph = sess.graph;
            var output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), new string[] { final_tensor_name });
            File.WriteAllBytes(graph_file_name, output_graph_def.ToByteArray());
        }
    }
}
