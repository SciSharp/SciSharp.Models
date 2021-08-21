using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models
{
    public class TrainingOptions
    {
        /// <summary>
        /// Training data
        /// </summary>
        public NDArray Data { get; set; }
        public TrainingOptions()
        {

        }
    }
}
