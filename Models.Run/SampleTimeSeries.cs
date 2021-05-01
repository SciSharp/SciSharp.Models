using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Models.Run
{
    /// <summary>
    /// https://www.tensorflow.org/tutorials/structured_data/time_series
    /// </summary>
    public class SampleTimeSeries
    {
        public void Run()
        {
            var zip_path = keras.utils.get_file("jena_climate_2009_2016.csv.zip",
                "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
                cache_subdir: "jena_climate_2009_2016",
                extract: true);
        }
    }
}
