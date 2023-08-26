using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow.Keras.Layers;

namespace Models.Run
{
    /// <summary>
    /// https://keras.io/examples/nlp/text_classification_with_transformer/
    /// </summary>
    public class SampleTransformer
    {
        public void Run()
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            TransformerClassification.Train(null);
            stopwatch.Stop();
            TimeSpan elapsedTime = stopwatch.Elapsed;
            Console.WriteLine("Elapsed time: {0} seconds", elapsedTime.TotalSeconds);
        }
    }
}
