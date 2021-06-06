using System;

namespace Models.Run
{
    class Program
    {
        static void Main(string[] args)
        {
            /*var tc = new SampleBinaryTextClassification();
            tc.Run();

            var ar = new SampleAudioRecognition();
            ar.LoadModel("simple_audio_model");
            ar.Run();*/

            var ts = new SampleTimeSeries();
            ts.Run();

            var yolo3 = new SampleYOLOv3();
            yolo3.Run();

            // var transformer = new SampleTransformer();
            // transformer.Run();
            Console.WriteLine("YOLOv3 is completed.");
            Console.ReadLine();
        }
    }
}
