using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace SciSharp.Models.ImageClassification
{
    public partial class CNN : IImageClassificationTask
    {
        public void Config(TaskOptions options)
        {
            throw new NotImplementedException();
        }

        public ModelPredictResult Predict(Tensor input)
        {
            throw new NotImplementedException();
        }

        public ModelTestResult Test()
        {
            throw new NotImplementedException();
        }

        public void Train(TrainingOptions options)
        {
            throw new NotImplementedException();
        }
    }
}
