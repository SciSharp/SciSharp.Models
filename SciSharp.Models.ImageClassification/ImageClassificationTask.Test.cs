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
        public ModelTestResult Test()
        {
            if (!File.Exists(output_graph))
                throw new FreezedGraphNotFoundException();

            using var graph = new Graph();
            graph.Import(output_graph);
            var (jpeg_data_tensor, decoded_image_tensor) = add_jpeg_decoding();

            using var sess = tf.Session(graph);

            var (test_accuracy, predictions) = run_final_eval(sess, null, class_count, image_dataset,
                           jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                           bottleneck_tensor);

            return new ModelTestResult
            {
                Accuracy = test_accuracy,
                Predictions = predictions
            };
        }
    }
}
