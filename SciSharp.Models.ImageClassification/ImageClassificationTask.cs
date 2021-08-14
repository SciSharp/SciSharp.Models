using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Keras.Utils;
using System.Linq;

namespace SciSharp.Models.ImageClassification
{
    public partial class ImageClassificationTask : IModelTask 
    {
        string _modelDir;
        string summaries_dir;
        string bottleneck_dir;
        string image_dir;
        bool isImportingGraph = true;
        ModelOptions _options;
        string output_graph;
        string[] labels;
        string output_label_path;

        public ImageClassificationTask(ModelOptions options)
        {
            _options = options;
            tf.compat.v1.disable_eager_execution();
            _modelDir = Path.Combine(Directory.GetCurrentDirectory(), "image_classification_v1");
            CHECKPOINT_NAME = Path.Join(_modelDir, "_retrain_checkpoint");
            output_graph = Path.Join(_modelDir, "output_graph.pb");
            output_label_path = Path.Join(_modelDir, "output_labels.txt");

            // download graph meta data
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/InceptionV3.meta";
            Web.Download(url, "graph", "InceptionV3.meta");

            // download variables.data checkpoint file.
            url = "https://github.com/SciSharp/TensorFlow.NET/raw/master/data/tfhub_modules.zip";
            Web.Download(url, _modelDir, "tfhub_modules.zip");
            Compress.UnZip(Path.Join(_modelDir, "tfhub_modules.zip"), "tfhub_modules");

            // Prepare necessary directories that can be used during training
            summaries_dir = Path.Join(_modelDir, "retrain_logs");
            Directory.CreateDirectory(summaries_dir);
            bottleneck_dir = Path.Join(_modelDir, "bottleneck");
            Directory.CreateDirectory(bottleneck_dir);

            image_dir = options.DataDir;
        }
    }
}
