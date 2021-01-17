using System;

namespace Models.Run
{
    class Program
    {
        static void Main(string[] args)
        {
            var yolo3 = new SampleYOLOv3();
            yolo3.Run();

            // var transformer = new SampleTransformer();
            // transformer.Run();
            Console.WriteLine("YOLOv3 is completed.");
            Console.ReadLine();
        }
    }
}
