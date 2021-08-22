using System;
using System.Collections.Generic;
using System.Text;

namespace SciSharp.Models.ImageClassification
{
    /// <summary>
    /// Network configuration
    /// </summary>
    public class ConvArgs
    {
        /// <summary>
        /// 1st Convolutional Layer
        /// Convolution filters are 5 x 5 pixels.
        /// </summary>
        public int FilterSize1 { get; set; } = 5;

        /// <summary>
        /// There are 16 of these filters.
        /// </summary>
        public int NumberOfFilters1 { get; set; } = 16;
        /// <summary>
        /// The stride of the sliding window
        /// </summary>
        public int Stride1 { get; set; } = 1;

        /// <summary>
        /// 2nd Convolutional Layer
        /// Convolution filters are 5 x 5 pixels.
        /// </summary>
        public int FilterSize2 { get; set; } = 5;

        /// <summary>
        /// There are 32 of these filters.
        /// </summary>
        public int NumberOfFilters2 { get; set; } = 32;

        /// <summary>
        /// The stride of the sliding window
        /// </summary>
        public int Stride2 { get; set; } = 1;

        /// <summary>
        /// Fully-connected layer.
        /// Number of neurons in fully-connected layer.
        /// </summary>
        public int NumberOfNeurons { get; set; } = 128;
    }
}
