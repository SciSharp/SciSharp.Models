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
    public partial class ImageClassificationTask 
    {
        public ModelPredictResult Predict(NDArray data)
        {
            if (!File.Exists(output_graph))
                throw new FreezedGraphNotFoundException();

            if (labels == null)
                labels = File.ReadAllLines(output_label_path);

            // import graph and variables
            using var graph = new Graph();
            graph.Import(output_graph, "");
            Tensor input = graph.OperationByName(input_tensor_name);
            Tensor output = graph.OperationByName(final_tensor_name);

            using var sess = tf.Session(graph);
            var result = sess.run(output, (input, data));
            var prob = np.squeeze(result);
            var idx = np.argmax(prob);

            return new ModelPredictResult
            {
                Probabilities = prob,
                Predictions = idx
            };
        }
    }
}
