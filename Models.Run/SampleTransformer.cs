using SciSharp.Models.Transformer;
using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.KerasApi;

namespace Models.Run
{
    /// <summary>
    /// https://keras.io/examples/nlp/text_classification_with_transformer/
    /// </summary>
    public class SampleTransformer
    {
        public void Run()
        {
            Transformer.Train(null);
        }
    }
}
