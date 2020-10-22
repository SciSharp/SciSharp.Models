using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models.YOLOv3
{
    public class BatchFeedingImage
    {
        public NDArray Image { get; set; }
        public List<LabelBorderBox> Targets { get; set; }
    }
}
