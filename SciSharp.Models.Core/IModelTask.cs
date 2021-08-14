using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models
{
    public interface IModelTask
    {
        void LoadData();
        void Train(TrainingOptions options);
        ModelTestResult Test();
        void SaveModel();
        ModelPredictResult Predict(NDArray data);
    }
}
