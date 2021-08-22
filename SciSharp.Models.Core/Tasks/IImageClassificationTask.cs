using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace SciSharp.Models
{
    public interface IImageClassificationTask
    {
        void Train(TrainingOptions options);
        void SetModelArgs<T>(T args);
        ModelTestResult Test();
        ModelPredictResult Predict(Tensor input);
        void Config(TaskOptions options);
    }
}
