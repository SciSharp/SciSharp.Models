using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Tensorflow;

namespace SciSharp.Models
{
    public class TaskOptions
    {
        /// <summary>
        /// The path of model
        /// </summary>
        /// <value></value>
        public string ModelPath { get; set; }
        public string WeightsPath { get; set; }
        public string LabelPath { get; set; }
        public string DataDir { get; set; }
        public float TestingPercentage { get; set; } = 0.2f;
        public float ValidationPercentage { get; set; } = 0.1f;

        /// <summary>
        /// The shape of input data
        /// if is a image, the shape should be (height, width, channel)
        /// the channel is 1 for gray image, 3 for RGB image
        /// </summary>
        /// <value></value>
        public Shape InputShape { get; set; }

        /// <summary>
        /// The number of class
        /// if is a binary classification, the number of class should be 2
        /// if is a multi-class classification, the number of class should be more than 2
        /// if is a regression, the number of class should be 1
        /// </summary>
        /// <value></value>
        public int NumberOfClass { get; set; }

        public TaskOptions()
        {
        }
    }
}
