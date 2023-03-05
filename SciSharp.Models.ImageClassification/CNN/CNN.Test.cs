using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace SciSharp.Models.ImageClassification
{
    public partial class CNN
    {
        public ModelTestResult Test(TestingOptions options)
        {
            var result = new ModelTestResult();

            var graph = tf.Graph().as_default();
            graph.Import(_options.ModelPath);

            var sess = tf.Session(graph);

            Tensor loss = graph.OperationByName("Train/Loss/loss");
            Tensor accuracy = graph.OperationByName("Train/Accuracy/accuracy");
            Tensor x = graph.OperationByName("Input/X");
            Tensor y = graph.OperationByName("Input/Y");

            (result.Loss, result.Accuracy) = sess.run((loss, accuracy), (x, options.TestingData.Features), (y, options.TestingData.Labels));
            print("---------------------------------------------------------");
            print($"Testing result: loss={result.Loss:0.0000}, accuracy={result.Accuracy:P}");
            print("---------------------------------------------------------");

            return result;
        }
    }
}
