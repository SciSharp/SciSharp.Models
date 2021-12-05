using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models.ObjectDetection
{
    public class LabelBorderBox
    {
        public NDArray Label { get; set; }
        public NDArray BorderBox { get; set; }
    }
}
