using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;
using Tensorflow.Keras.Utils;
using System.Linq;

namespace SciSharp.Models.ImageClassification
{
    /// <summary>
    /// In this tutorial, we will reuse the feature extraction capabilities from powerful image classifiers trained on ImageNet 
    /// and simply train a new classification layer on top. Transfer learning is a technique that shortcuts much of this 
    /// by taking a piece of a model that has already been trained on a related task and reusing it in a new model.
    /// 
    /// https://www.tensorflow.org/hub/tutorials/image_retraining
    /// </summary>
    public partial class TransferLearning : IImageClassificationTask 
    {
        string taskDir;
        string summaries_dir;
        string bottleneck_dir;
        bool isImportingGraph = true;
        TaskOptions _options;
        string[] labels;
        
        public TransferLearning()
        {
            taskDir = Path.Combine(Directory.GetCurrentDirectory(), "image_classification_v1");

            // download graph meta data
            var url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/InceptionV3.meta";
            Web.Download(url, "graph", "InceptionV3.meta");

            // download variables.data checkpoint file.
            url = "https://github.com/SciSharp/TensorFlow.NET/raw/master/data/tfhub_modules.zip";
            Web.Download(url, taskDir, "tfhub_modules.zip");
            Compress.UnZip(Path.Combine(taskDir, "tfhub_modules.zip"), "tfhub_modules");

            // Prepare necessary directories that can be used during training
            summaries_dir = Path.Combine(taskDir, "retrain_logs");
            Directory.CreateDirectory(summaries_dir);
            bottleneck_dir = Path.Combine(taskDir, "bottleneck");
            Directory.CreateDirectory(bottleneck_dir);
        }

        public void Config(TaskOptions options)
        {
            options.ModelPath = options.ModelPath ?? Path.Combine(taskDir, "saved_model.pb");
            options.LabelPath = options.LabelPath ?? Path.Combine(taskDir, "labels.txt");
            _options = options;
        }

        public void SetModelArgs<T>(T args)
        {
            
        }
    }
}
