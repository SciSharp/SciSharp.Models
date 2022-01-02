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
            //var sample = new SampleBinaryTextClassification();

            //var ar = new SampleAudioRecognition();
            //ar.LoadModel("simple_audio_model");

            var sample = new SampleTimeSeries();

            sample.Run();

            // var transformer = new SampleTransformer();
            // transformer.Run();
        }
    }
}
