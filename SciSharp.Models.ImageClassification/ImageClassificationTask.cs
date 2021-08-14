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
        string taskDir;
        string summaries_dir;
        string bottleneck_dir;
        string image_dir;
        bool isImportingGraph = true;
        TaskOptions options;
        TrainingOptions trainingOptions;
        string[] labels;
        
        public ImageClassificationTask()
        {
            tf.compat.v1.disable_eager_execution();
            taskDir = Path.Combine(Directory.GetCurrentDirectory(), "image_classification_v1");

            // download graph meta data
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/InceptionV3.meta";
            Web.Download(url, "graph", "InceptionV3.meta");

            // download variables.data checkpoint file.
            url = "https://github.com/SciSharp/TensorFlow.NET/raw/master/data/tfhub_modules.zip";
            Web.Download(url, taskDir, "tfhub_modules.zip");
            Compress.UnZip(Path.Join(taskDir, "tfhub_modules.zip"), "tfhub_modules");

            // Prepare necessary directories that can be used during training
            summaries_dir = Path.Join(taskDir, "retrain_logs");
            Directory.CreateDirectory(summaries_dir);
            bottleneck_dir = Path.Join(taskDir, "bottleneck");
            Directory.CreateDirectory(bottleneck_dir);
        }
    }
}
