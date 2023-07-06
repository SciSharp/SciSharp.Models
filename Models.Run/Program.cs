using SciSharp.Models;
using System;

namespace Models.Run
{
    class Program
    {
        static void Main(string[] args)
        {
            Run();
        }

        static void Run()
        {
            //var wp = new WeatherPrediction();
            //wp.Run();
            var st = new SampleTransformer();
            st.Run();
        }
    }
}
