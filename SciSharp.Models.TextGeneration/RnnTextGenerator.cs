using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace SciSharp.Models.TextClassification
{
    public class RnnTextGenerator : ITextGenerationTask
    {
        public void Config(TaskOptions options)
        {
            throw new NotImplementedException();
        }

        public ModelPredictResult Predict(Tensor input)
        {
            throw new NotImplementedException();
        }

        public void Run()
        {

        }

        public void SetModelArgs<T>(T args)
        {
            throw new NotImplementedException();
        }

        public ModelTestResult Test(TestingOptions options)
        {
            throw new NotImplementedException();
        }

        public void Train(TrainingOptions options)
        {
            throw new NotImplementedException();
        }
    }
}
