using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace SciSharp.Models.ImageClassification
{
    public partial class CNN
    {
        (NDArray, NDArray) GetNextBatch(NDArray x, NDArray y, int start, int end)
        {
            var slice = new Slice(start, end);
            var x_batch = x[slice];
            var y_batch = y[slice];
            return (x_batch, y_batch);
        }

        (NDArray, NDArray) Randomize(NDArray x, NDArray y)
        {
            var perm = np.random.permutation(len(y));
            np.random.shuffle(perm);
            return (x[perm], y[perm]);
        }
    }
}
