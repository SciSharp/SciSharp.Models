using SciSharp.Models;
using System;

namespace Models.Run
{
    class Program
    {
        static void Main(string[] args)
        {
            Run();

            GC.Collect();
            GC.WaitForPendingFinalizers();

            Console.WriteLine("Completed.");
            Console.ReadLine();
        }

        static void Run()
        {
            /*var tc = new SampleBinaryTextClassification();
            tc.Run();

            var ar = new SampleAudioRecognition();
            ar.LoadModel("simple_audio_model");
            ar.Run();*/

            /*var ts = new SampleTimeSeries();
            ts.Run();*/

            var yolo3 = new SampleYOLOv3();
            yolo3.Run();

            // var transformer = new SampleTransformer();
            // transformer.Run();
        }
    }
}
