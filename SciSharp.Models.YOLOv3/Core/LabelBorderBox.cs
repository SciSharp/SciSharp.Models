using NumSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models.YOLOv3
{
    public class LabelBorderBox
    {
        public NDArray Label { get; set; }
        public NDArray BorderBox { get; set; }
    }
}
