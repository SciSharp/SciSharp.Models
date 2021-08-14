using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using static Tensorflow.Binding;
using System.Linq;

namespace SciSharp.Models.ImageClassification
{
    public partial class ImageClassificationTask 
    {
        public void LoadData()
        {
            // Look at the folder structure, and create lists of all the images.
            image_dataset = CreateDatasetFromDirectory();
            class_count = len(image_dataset);
            if (class_count == 0)
                print($"No valid folders of images found at {image_dir}");
            if (class_count == 1)
                print("Only one valid folder of images found at " +
                     image_dir +
                     " - multiple classes are needed for classification.");
        }

        /// <summary>
        /// Builds a list of training images from the file system.
        /// </summary>
        Dictionary<string, Dictionary<string, string[]>> CreateDatasetFromDirectory()
        {
            var sub_dirs = tf.gfile.Walk(image_dir)
                .Select(x => x.Item1)
                .OrderBy(x => x)
                .ToArray();

            var result = new Dictionary<string, Dictionary<string, string[]>>();

            foreach (var sub_dir in sub_dirs)
            {
                var dir_name = sub_dir.Split(Path.DirectorySeparatorChar).Last();
                print($"Looking for images in '{dir_name}'");
                var file_list = Directory.GetFiles(sub_dir);
                if (len(file_list) < 20)
                    print($"WARNING: Folder has less than 20 images, which may cause issues.");

                var label_name = dir_name.ToLower();
                result[label_name] = new Dictionary<string, string[]>();
                int testing_count = (int)Math.Floor(file_list.Length * _options.TrainingOptions.TestingPercentage);
                int validation_count = (int)Math.Floor(file_list.Length * _options.TrainingOptions.ValidationPercentage);
                result[label_name]["testing"] = file_list.Take(testing_count).ToArray();
                result[label_name]["validation"] = file_list.Skip(testing_count).Take(validation_count).ToArray();
                result[label_name]["training"] = file_list.Skip(testing_count + validation_count).ToArray();
            }

            return result;
        }
    }
}
