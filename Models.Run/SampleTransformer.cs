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
        public bool Run()
        {
            var vocab_size = 20000; // Only consider the top 20k words
            var maxlen = 200; // Only consider the first 200 words of each movie review
            var ((x_train, y_train), (x_val, y_val)) = keras.datasets.imdb.load_data(num_words: vocab_size);
            return true;
        }
    }
}
