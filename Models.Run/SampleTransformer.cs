using SciSharp.Models.Transformer;
using System;
using System.Diagnostics;
using Tensorflow.Keras.ArgsDefinition;
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
            var transformer = new TransformerClassification(new TransformerClassificationArgs
            {
            });

            var model = transformer.Train();
            transformer.Save(model, "");
            var model = transformer.Load("");
            transformer.Evaluate(model);

            stopwatch.Stop();
            TimeSpan elapsedTime = stopwatch.Elapsed;
            Console.WriteLine("Elapsed time: {0} seconds", elapsedTime.TotalSeconds);
        }
    }
}
