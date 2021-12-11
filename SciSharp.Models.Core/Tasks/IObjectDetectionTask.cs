using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace SciSharp.Models
{
    public interface IObjectDetectionTask
    {
        void Train(TrainingOptions options);
        void SetModelArgs<T>(T args);
        ModelTestResult Test(TestingOptions options);
        ModelPredictResult Predict(Tensor input);
        void Config(TaskOptions options);
    }
}
