using NumSharp;
using SciSharp.Models.YOLOv3;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace TensorFlowNET.Examples.ImageProcessing.YOLO
{
    /// <summary>
    /// Implementation of YOLO v3 object detector in Tensorflow
    /// https://github.com/YunYang1994/tensorflow-yolov3
    /// </summary>
    public class SampleYOLOv3
    {
        YOLOv3 yolo;
        YoloDataset trainset, testset;

        YoloConfig cfg;

        Tensor input_tensor;

        Model model;

        public bool Run()
        {
            tf.enable_eager_execution();

            cfg = new YoloConfig("YOLOv3");
            yolo = new YOLOv3(cfg);

            PrepareData();
            Train();

            return true;
        }

        /// <summary>
        /// Train model in batch image
        /// </summary>
        /// <param name="image_data"></param>
        /// <param name="targets"></param>
        void TrainStep(NDArray image_data, List<LabelBorderBox> targets)
        {
            using var tape = tf.GradientTape();
            var pred_result = model.Apply(image_data, is_training: true);
            var giou_loss = tf.constant(0.0f);
            var conf_loss = tf.constant(0.0f);
            var prob_loss = tf.constant(0.0f);

            // optimizing process in different border boxes.
            foreach(var (i, target) in enumerate(targets))
            {
                var (conv, pred) = (pred_result[i * 2], pred_result[i * 2 + 1]);
                var loss_items = yolo.compute_loss(pred, conv, target.Label, target.BorderBox , i);
                giou_loss += loss_items[0];
                conf_loss += loss_items[1];
                prob_loss += loss_items[2];
            }

            var total_loss = giou_loss + conf_loss + prob_loss;

            var gradients = tape.gradient(total_loss, model.trainable_variables);
        }

        public void Train()
        {
            input_tensor = keras.layers.Input((416, 416, 3));
            
            var conv_tensors = yolo.Apply(input_tensor);

            var output_tensors = new List<Tensor>();
            foreach (var (i, conv_tensor) in enumerate(conv_tensors))
            {
                var pred_tensor = yolo.Decode(conv_tensor, i);
                output_tensors.append(conv_tensor);
                output_tensors.append(pred_tensor);
            }

            model = keras.Model(input_tensor, output_tensors);
            // model.summary();
            // model.load_weights("./yolov3");

            var optimizer = keras.optimizers.Adam();
            foreach (var epoch in range(cfg.TRAIN.EPOCHS))
            {
                // tf.print('EPOCH %3d' % (epoch + 1))
                foreach (var dataset in trainset)
                    TrainStep(dataset.Image, dataset.Targets);
            }
        }

        public void PrepareData()
        {
            string dataDir = Path.Combine("YOLOv3", "data");
            Directory.CreateDirectory(dataDir);

            trainset = new YoloDataset("train", cfg);
            testset = new YoloDataset("test", cfg);
        }
    }
}
