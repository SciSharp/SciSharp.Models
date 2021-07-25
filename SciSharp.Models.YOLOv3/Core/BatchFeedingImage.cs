using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow.NumPy;

namespace SciSharp.Models.YOLOv3
{
    public class BatchFeedingImage
    {
        public NDArray Image { get; set; }
        public List<LabelBorderBox> Targets { get; set; }
    }
}
