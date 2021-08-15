using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models
{
    public interface IImageClassificationTask
    {
        void Train(TrainingOptions options);
        ModelTestResult Test();
        ModelPredictResult Predict(string imagePath);
        void Config(TaskOptions options);
    }
}
