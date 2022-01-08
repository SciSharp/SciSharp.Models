using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace SciSharp.Models
{
    public interface ITimeSeriesTask
    {
        (IDatasetV2, IDatasetV2, IDatasetV2) GenerateDataset<TDataSource>(Func<TDataSource> ds);
        void Train(TrainingOptions options);
        void SetModelArgs<T>(T args);
        float Test(TestingOptions options);
        Tensor Predict(Tensor input);
        void Config(TaskOptions options);
    }
}
