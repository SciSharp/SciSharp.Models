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
        /// <summary>
        /// Prediction
        /// labels mapping, it's from output_lables.txt
        /// 0 - daisy
        /// 1 - dandelion
        /// 2 - roses
        /// 3 - sunflowers
        /// 4 - tulips
        /// </summary>
        /// <param name="sess_"></param>
        public ModelPredictResult Predict()
        {
            if (!File.Exists(output_graph))
                throw new FreezedGraphNotFoundException();

            if (labels == null)
                labels = File.ReadAllLines(output_label_path);

            // predict image
            var img_path = Path.Join(image_dir, "daisy", "5547758_eea9edfd54_n.jpg");
            var fileBytes = ReadTensorFromImageFile(img_path);

            // import graph and variables
            using var graph = new Graph();
            graph.Import(output_graph, "");
            Tensor input = graph.OperationByName(input_tensor_name);
            Tensor output = graph.OperationByName(final_tensor_name);

            using var sess = tf.Session(graph);
            var result = sess.run(output, (input, fileBytes));
            var prob = np.squeeze(result);
            var idx = np.argmax(prob);
            print($"Prediction result: [{labels[idx]} {prob[idx]}] for {img_path}.");

            return new ModelPredictResult
            {
                Probabilities = prob,
                Predictions = idx
            };
        }
    }
}
