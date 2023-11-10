using SciSharp.Models.ImageClassification;
using SciSharp.Models.ImageClassification.Zoo;
using SciSharp.Models.TextClassification;
using System;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Models.Run
{
    internal class SampleFolderImageClassifcation
    {
        private static (FolderClassificationConfig config, int img_size) GetConfig()
        {
            var config = new FolderClassificationConfig();
            if (Environment.OSVersion.Platform == PlatformID.Win32NT)
                config.BaseFolder = "D:\\data\\flower_photos";
            else
                config.BaseFolder = "/mnt/d/data/flower_photos";

            config.DataDir = "";
            var img_size = 224;
            config.InputShape = (img_size, img_size);

            return (config, img_size);
        }

        public void RunTrain()
        {
            var (config, img_size) = GetConfig();

            config.BatchSize = 24;
            config.ValidationStep = 5;
            config.Epoch = 20;

            var model = new AlexNet();
            // var model = new DenseNet();
            // var model = new GoogLeNet();
            // var model = new MobilenetV2();
            // var model = new NiN();
            // var model = new ResNet();
            // var model = new VGG();

            var classifier = new FolderClassification(config, model);

            config.WeightsPath = $"{model.GetType().Name}_{img_size}x{img_size}_weights.ckpt";

            classifier.Train();
        }

        public void RunPredictFolder()
        {
            var (config, img_size) = GetConfig();

            var model = new DenseNet();
            //var model = new AlexNet();

            var classifier = new FolderClassification(config, model);

            config.WeightsPath = $"{model.GetType().Name}_{img_size}x{img_size}_weights.ckpt";

            var imageFile = Path.Combine(config.BaseFolder, "roses", "160954292_6c2b4fda65_n.jpg");
            var result = classifier.Predict(imageFile);
            Console.Write($"{result.Label}({result.Probability})");
        }


        public void RunValidateFolder()
        {
            var (config, img_size) = GetConfig();
            var model = new AlexNet();
            var classifier = new FolderClassification(config, model);

            config.WeightsPath = $"{model.GetType().Name}_{img_size}x{img_size}_weights.ckpt";

            var roseFolder = Path.Combine(config.BaseFolder, "roses");

            var files = Directory.GetFiles(roseFolder);
            var results = classifier.Predict(files);

            var correct = results.Where(r => r.Label == "roses").Count();
            var correct80 = results.Where(r => r.Label == "roses" && r.Probability > 80).Count();
            var total = results.Count();
            Console.WriteLine($"Correct: {correct}/{total}({(double)correct / total}) ");
            Console.WriteLine($"Correct80: {correct80}/{total}({(double)correct80 / total}) ");
        }
    }
}
