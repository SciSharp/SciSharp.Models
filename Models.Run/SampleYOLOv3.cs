using SciSharp.Models.YOLOv3;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static SharpCV.Binding;
using SharpCV;
using Utils = SciSharp.Models.YOLOv3.Utils;
using Tensorflow.NumPy;
using System.Diagnostics;

namespace Models.Run
{
    /// <summary>
    /// Implementation of YOLO v3 object detector in Tensorflow
    /// https://github.com/YunYang1994/tensorflow-yolov3
    /// </summary>
    public class SampleYOLOv3
    {
        YOLOv3 yolo;
        YoloConfig cfg;

        OptimizerV2 optimizer;
        int global_steps;
        int warmup_steps;
        int total_steps;

        public bool Run()
        {
            // tf.enable_eager_execution();
            // tf.debugging.set_log_device_placement(true);
            cfg = new YoloConfig("YOLOv3");
            yolo = new YOLOv3(cfg);

            var (trainset, testset) = PrepareData();
            Train(trainset);
            // Test(testset);

            return true;
        }

        /// <summary>
        /// Train model in batch image
        /// </summary>
        /// <param name="image_data"></param>
        /// <param name="targets"></param>
        Tensor TrainStep(Model model, NDArray image_data, List<LabelBorderBox> targets)
        {
            using var tape = tf.GradientTape();
            var pred_result = model.Apply(image_data, training: true);

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

            GC.Collect();
            GC.WaitForPendingFinalizers();

            var gradients = tape.gradient(total_loss, model.trainable_variables);
            optimizer.apply_gradients(zip(gradients, model.trainable_variables.Select(x => x as ResourceVariable)));

            float lr = optimizer.lr.numpy();
            print($"=> STEP {global_steps:D4} lr:{lr} giou_loss: {giou_loss.numpy()} conf_loss: {conf_loss.numpy()} prob_loss: {prob_loss.numpy()} total_loss: {total_loss.numpy()}");
            global_steps++;

            // update learning rate
            if (global_steps < warmup_steps)
            {
                lr = global_steps / (warmup_steps + 0f) * cfg.TRAIN.LEARN_RATE_INIT;
            }
            else
            {
                lr = (cfg.TRAIN.LEARN_RATE_END + 0.5f * (cfg.TRAIN.LEARN_RATE_INIT - cfg.TRAIN.LEARN_RATE_END) * 
                    (1 + tf.cos((global_steps - warmup_steps + 0f) / (total_steps - warmup_steps) * (float)np.pi))).numpy();
            }
            var lr_tensor = tf.constant(lr);
            optimizer.lr.assign(lr_tensor);

            return total_loss;
        }

        public void Train(YoloDataset trainset)
        {
            var input_layer = keras.layers.Input((416, 416, 3));
            var conv_tensors = yolo.Apply(input_layer);

            var output_tensors = new Tensors();
            foreach(var (i, conv_tensor) in enumerate(conv_tensors))
            {
                var pred_tensor = yolo.Decode(conv_tensor, i);
                output_tensors.Add(conv_tensor);
                output_tensors.Add(pred_tensor);
            }

            Model model = keras.Model(input_layer, output_tensors);
            model.summary();

            // download wights from https://drive.google.com/file/d/1J5N5Pqf1BG1sN_GWDzgViBcdK2757-tS/view?usp=sharing
            // model.load_weights("./YOLOv3/yolov3.h5");

            optimizer = keras.optimizers.Adam();
            int steps_per_epoch = trainset.Length;
            total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch;
            warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch;

            float loss = 1000;
            foreach (var epoch in range(cfg.TRAIN.EPOCHS))
            {
                print($"EPOCH {epoch + 1:D4}");
                float current_loss = -1;
                foreach (var dataset in trainset)
                {
                    var watch = new Stopwatch();
                    watch.Start();
                    current_loss = TrainStep(model, dataset.Image, dataset.Targets).numpy();
                    Console.WriteLine($"spent {watch.ElapsedMilliseconds} ms.");
                }
                if(current_loss < loss)
                {
                    loss = current_loss;
                    model.save_weights($"./YOLOv3/yolov3.{loss:F2}.h5");
                }
            }
        }

        public void Test(YoloDataset testset)
        {
            var input_layer = keras.layers.Input((cfg.TEST.INPUT_SIZE[0], cfg.TEST.INPUT_SIZE[0], 3));
            var feature_maps = yolo.Apply(input_layer);

            var bbox_tensors = new Tensors();
            foreach (var (i, fm) in enumerate(feature_maps))
            {
                var bbox_tensor = yolo.Decode(fm, i);
                bbox_tensors.Add(bbox_tensor);
            }
            Model model = keras.Model(input_layer, bbox_tensors);

            // var weights = model.load_weights("D:/Projects/SciSharp.Models/yolov3.h5");
            model.load_weights("./YOLOv3/yolov3.mnist.h5");

            var mAP_dir = Path.Combine("mAP", "ground-truth");
            Directory.CreateDirectory(mAP_dir);
            
            var annotation_files = File.ReadAllLines(cfg.TEST.ANNOT_PATH);
            foreach (var (num, line) in enumerate(annotation_files))
            {
                var annotation = line.Split(' ');
                var image_path = annotation[0];
                var image_name = image_path.Split(Path.DirectorySeparatorChar).Last();
                var original_image = cv2.imread(image_path);
                var image = cv2.cvtColor(original_image, ColorConversionCodes.COLOR_BGR2RGB);
                var count = annotation.Skip(1).Count();
                var bbox_data_gt = np.zeros((count, 5), np.int32);
                foreach (var (i, box) in enumerate(annotation.Skip(1)))
                {
                    bbox_data_gt[i] = np.array(box.Split(',').Select(x => int.Parse(x)).ToArray());
                };
                var (bboxes_gt, classes_gt) = (bbox_data_gt[":", ":4"], bbox_data_gt[":", "4"]);
                
                print($"=> ground truth of {image_name}:");

                var bbox_mess_file = new List<string>();
                foreach (var i in range((int)bboxes_gt.shape[0]))
                {
                    var class_name = yolo.Classes[classes_gt[i]];
                    var bbox_mess = $"{class_name} {string.Join(" ", bboxes_gt[i].ToArray<int>())}";
                    bbox_mess_file.Add(bbox_mess);
                    print('\t' + bbox_mess);
                }

                var ground_truth_path = Path.Combine(mAP_dir, $"{num}.txt");
                File.WriteAllLines(ground_truth_path, bbox_mess_file);
                print($"=> predict result of {image_name}:");
                // Predict Process
                var image_size = image.shape.as_int_list().Take(2).ToArray();
                var image_data = Utils.image_preporcess(image, image_size).Item1;

                image_data = image_data[np.newaxis, Slice.Ellipsis];
                var pred_bbox = model.predict(image_data);
                pred_bbox = pred_bbox.Select(x => tf.reshape(x, new object[] { -1, tf.shape(x)[-1] })).ToList();
                var pred_bbox_concat = tf.concat(pred_bbox, axis: 0);
                var bboxes = Utils.postprocess_boxes(pred_bbox_concat.numpy(), image_size, cfg.TEST.INPUT_SIZE[0], cfg.TEST.SCORE_THRESHOLD);
                if(bboxes.size > 0)
                {
                    var best_box_results = Utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method: "nms");
                    Utils.draw_bbox(image, best_box_results, yolo.Classes.Values.ToArray());
                    cv2.imwrite(Path.Combine(cfg.TEST.DECTECTED_IMAGE_PATH, Path.GetFileName(image_name)), image);
                }
            }
        }

        public (YoloDataset, YoloDataset) PrepareData()
        {
            string dataDir = Path.Combine("YOLOv3", "data");
            Directory.CreateDirectory(dataDir);

            var trainset = new YoloDataset("train", cfg);
            var testset = new YoloDataset("test", cfg);
            return (trainset, testset);
        }
    }
}
