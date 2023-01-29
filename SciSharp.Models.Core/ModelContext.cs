using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models
{
    public class ModelContext
    {
        public IImageClassificationTask ImageClassificationTask { get; set; }
        public IObjectDetectionTask ObjectDetectionTask { get; set; }
        public ITimeSeriesTask TimeSeriesTask { get; set; }
        public ITextGenerationTask TextGenerationTask { get; set; }
    }
}
