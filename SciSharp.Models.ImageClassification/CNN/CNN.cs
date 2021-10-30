using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace SciSharp.Models.ImageClassification
{
    public partial class CNN : IImageClassificationTask
    {
        string _taskDir;
        TaskOptions _options;
        int display_freq = 100;
        ConvArgs _convArgs;
        string[] labels;

        public CNN()
        {
            tf.compat.v1.disable_eager_execution();
            _taskDir = Path.Combine(Directory.GetCurrentDirectory(), "image_classification_cnn_v1");
        }

        public void Config(TaskOptions options)
        {
            _options = options;
        }

        public void SetModelArgs<T>(T args)
        {
            _convArgs = (ConvArgs)Convert.ChangeType(args, typeof(ConvArgs));
        }
    }
}
