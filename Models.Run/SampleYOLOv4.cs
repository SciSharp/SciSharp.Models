using System;
using System.Collections.Generic;
using System.Text;
using static Tensorflow.Binding;

namespace Models.Run
{
    /// <summary>
    /// YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.0.
    /// https://github.com/hunglc007/tensorflow-yolov4-tflite
    /// </summary>
    public class SampleYOLOv4
    {
        public bool Run()
        {
            tf.enable_eager_execution();

            /*cfg = new YoloConfig("YOLOv4");
            yolo = new YOLOv4(cfg);

            PrepareData();
            Train();*/

            return true;
        }
    }
}
