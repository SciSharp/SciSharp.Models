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
    public partial class TransferLearning 
    {
        void SaveModel()
        {
            // Write out the trained graph and labels with the weights stored as
            // constants.
            print($"Saving final result to: {_options.ModelPath}");
            var (sess, _, _, _, _, _) = build_eval_session();
            var graph = sess.graph;
            var output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), new string[] { final_tensor_name });
            File.WriteAllBytes(_options.ModelPath, output_graph_def.ToByteArray());
            File.WriteAllText(_options.LabelPath, string.Join("\n", image_dataset.Keys));
        }
    }
}
