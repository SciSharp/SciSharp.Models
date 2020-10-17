using System;
using TensorFlowNET.Examples.ImageProcessing.YOLO;

namespace Models.Run
{
    class Program
    {
        static void Main(string[] args)
        {
            var yolo3 = new SampleYOLOv3();
            yolo3.Run();
        }
    }
}
