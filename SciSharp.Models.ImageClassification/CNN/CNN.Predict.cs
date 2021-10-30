using SciSharp.Models.Exceptions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace SciSharp.Models.ImageClassification
{
    public partial class CNN
    {
        Session predictSession;
        public ModelPredictResult Predict(Tensor input)
        {
            if (!File.Exists(_options.ModelPath))
                throw new FreezedGraphNotFoundException();

            if (labels == null)
                labels = File.ReadAllLines(_options.LabelPath);

            // import graph and variables
            if (predictSession == null)
            {
                var graph = tf.Graph().as_default();
                graph.Import(_options.ModelPath);
                predictSession = tf.Session(graph);
            }

            Tensor x = predictSession.graph.OperationByName("Input/X");
            // Tensor prediction = predictSession.graph.OperationByName("Train/Prediction/predictions");
            Tensor output = predictSession.graph.OperationByName("OUT/add");

            var result = predictSession.run(output, (x, input));

            var prob = np.squeeze(result);
            var idx = np.argmax(prob);

            print($"Predicted result: {labels[idx]} - {(float)prob[idx] / 100:P}");

            return new ModelPredictResult
            {
                Label = labels[idx],
                Probability = prob[idx]
            };
        }
    }
}
