using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;

namespace SciSharp.Models
{
    public class TestingOptions
    {
        /// <summary>
        /// Testing data
        /// </summary>
        public FeatureAndLabel TestingData { get; set; }
        public IDatasetV2 Dataset { get; set; }

        public TestingOptions()
        {

        }
    }
}
