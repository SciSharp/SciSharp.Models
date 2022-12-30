using System.Linq;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.IO;
using Tensorflow.NumPy;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using System.Diagnostics;
using System;

namespace SciSharp.Models.ObjectDetection
{
    // https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3
    public partial class YOLOv3 : IObjectDetectionTask
    {
        YOLOv3 yolo;
        YoloConfig cfg;
        public Dictionary<int, string> Classes { get; set; }
        int num_class => Classes.Count;
        int[] strides;
        NDArray anchors;
        int anchor_per_scale;
        float iou_loss_thresh;
        string upsample_method;
        Tensor conv;
        Tensor pred_sbbox;
        Tensor pred_mbbox;
        Tensor pred_lbbox;

        OptimizerV2 optimizer;
        int global_steps;
        int warmup_steps;
        int total_steps;
        YoloTrainingOptions _trainingOptions;

        public YOLOv3()
        {

        }

        public YOLOv3(YoloConfig cfg_)
        {
            cfg = cfg_;
            Classes = Utils.read_class_names(cfg.YOLO.CLASSES);
            strides = cfg.YOLO.STRIDES;
            anchors = Utils.get_anchors(cfg.YOLO.ANCHORS);
            anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE;
            iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH;
            upsample_method = cfg.YOLO.UPSAMPLE_METHOD;
        }

        public Tensor[] Apply(Tensor input_layer)
        {
            var (route_1, route_2, conv) = Backbone.darknet53(input_layer);

            conv = Common.convolutional(conv, (1, 1, 1024, 512));
            conv = Common.convolutional(conv, (3, 3, 512, 1024));
            conv = Common.convolutional(conv, (1, 1, 1024, 512));
            conv = Common.convolutional(conv, (3, 3, 512, 1024));
            conv = Common.convolutional(conv, (1, 1, 1024, 512));

            var conv_lobj_branch = Common.convolutional(conv, (3, 3, 512, 1024));
            var conv_lbbox = Common.convolutional(conv_lobj_branch, 
                (1, 1, 1024, 3 * (num_class + 5)), 
                activate: false, bn: false);

            conv = Common.convolutional(conv, (1, 1, 512, 256));
            conv = Common.upsample(conv);

            conv = keras.layers.Concatenate(axis: -1).Apply(new[] { conv, route_2 });

            conv = Common.convolutional(conv, (1, 1, 768, 256));
            conv = Common.convolutional(conv, (3, 3, 256, 512));
            conv = Common.convolutional(conv, (1, 1, 512, 256));
            conv = Common.convolutional(conv, (3, 3, 256, 512));
            conv = Common.convolutional(conv, (1, 1, 512, 256));

            var conv_mobj_branch = Common.convolutional(conv, (3, 3, 256, 512));
            var conv_mbbox = Common.convolutional(conv_mobj_branch, 
                (1, 1, 512, 3 * (num_class + 5)),
                activate: false, bn: false);

            conv = Common.convolutional(conv, (1, 1, 256, 128));
            conv = Common.upsample(conv);

            conv = keras.layers.Concatenate(axis: -1).Apply(new[] { conv, route_1 });

            conv = Common.convolutional(conv, (1, 1, 384, 128));
            conv = Common.convolutional(conv, (3, 3, 128, 256));
            conv = Common.convolutional(conv, (1, 1, 256, 128));
            conv = Common.convolutional(conv, (3, 3, 128, 256));
            conv = Common.convolutional(conv, (1, 1, 256, 128));

            var conv_sobj_branch = Common.convolutional(conv, (3, 3, 128, 256));
            var conv_sbbox = Common.convolutional(conv_sobj_branch,
                (1, 1, 256, 3 * (num_class + 5)),
                activate: false, bn: false);

            return new[] { conv_sbbox, conv_mbbox, conv_lbbox };
        }

        public Tensor Decode(Tensor conv_output, int i = 0)
        {
            var conv_shape = tf.shape(conv_output);
            var batch_size = conv_shape[0];
            var output_size = conv_shape[1];

            conv_output = tf.reshape(conv_output, new object[] { batch_size, output_size, output_size, 3, 5 + num_class });

            var conv_raw_dxdy = conv_output[":", ":", ":", ":", "0:2"];
            var conv_raw_dwdh = conv_output[":", ":", ":", ":", "2:4"];
            var conv_raw_conf = conv_output[":", ":", ":", ":", "4:5"];
            var conv_raw_prob = conv_output[":", ":", ":", ":", "5:"];

            var y = tf.tile(tf.range(output_size, dtype: tf.int32)[Slice.All, tf.newaxis], new object[] { 1, output_size });
            var x = tf.tile(tf.range(output_size, dtype: tf.int32)[tf.newaxis, Slice.All], new object[] { output_size, 1 });

            Tensor xy_grid = keras.layers.Concatenate(axis: -1).Apply(new[] { x[Slice.All, Slice.All, tf.newaxis], y[Slice.All, Slice.All, tf.newaxis] });
            xy_grid = tf.tile(xy_grid[tf.newaxis, Slice.All, Slice.All, tf.newaxis, Slice.All], new object[] { batch_size, 1, 1, 3, 1 });
            xy_grid = tf.cast(xy_grid, tf.float32);

            var pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i];
            var pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i];
            Tensor pred_xywh = keras.layers.Concatenate(axis: -1).Apply(new[] { pred_xy, pred_wh });
            var pred_conf = tf.sigmoid(conv_raw_conf);
            var pred_prob = tf.sigmoid(conv_raw_prob);
            
            return keras.layers.Concatenate(axis: -1).Apply(new[] { pred_xywh, pred_conf, pred_prob });
        }

        public Tensor[] compute_loss(Tensor pred, Tensor conv, NDArray label, NDArray bboxes, int i = 0)
        {
            var conv_shape = tf.shape(conv);
            var batch_size = conv_shape[0];
            var output_size = conv_shape[1];
            var input_size = strides[i] * output_size;
            conv = tf.reshape(conv, new object[] { batch_size, output_size, output_size, 3, 5 + num_class });

            var conv_raw_conf = conv[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(4, 5)];
            var conv_raw_prob = conv[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(5)];

            var pred_xywh = pred[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(0, 4)];
            var pred_conf = pred[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(4, 5)];

            var label_xywh = label[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(0, 4)];
            var respond_bbox = label[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(4, 5)];
            var label_prob = label[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(5)];

            var giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis: -1);
            input_size = tf.cast(input_size, tf.float32);

            var label_xywh_1 = label_xywh[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(2, 3)] 
                    * label_xywh[Slice.All, Slice.All, Slice.All, Slice.All, new Slice(3, 4)];
            var bbox_loss_scale = 2.0 - 1.0 * label_xywh_1 / (input_size * input_size);
            var giou_loss = respond_bbox * bbox_loss_scale * (1 - giou);

            var iou = bbox_iou(pred_xywh[Slice.All, Slice.All, Slice.All, Slice.All, np.newaxis, Slice.All], bboxes[Slice.All, np.newaxis, np.newaxis, np.newaxis, Slice.All, Slice.All]);
            var max_iou = tf.expand_dims(tf.reduce_max(iou, axis: -1), axis: -1);

            var respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < cfg.YOLO.IOU_LOSS_THRESH, tf.float32);

            var conf_focal = tf.pow(respond_bbox - pred_conf, 2);

            var sigmoid1 = tf.nn.sigmoid_cross_entropy_with_logits(labels: respond_bbox, logits: conv_raw_conf);
            var sigmoid2 = tf.nn.sigmoid_cross_entropy_with_logits(labels: respond_bbox, logits: conv_raw_conf);
            var conf_loss = conf_focal * (respond_bbox * sigmoid1 + respond_bgd * sigmoid2);

            var prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels: label_prob, logits: conv_raw_prob);
            
            giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis: new[] { 1, 2, 3, 4 }));
            conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis: new[] { 1, 2, 3, 4 }));
            prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis: new[] { 1, 2, 3, 4 }));

            return new[] { giou_loss, conf_loss, prob_loss };
        }

        public Tensor focal(Tensor target, Tensor actual, int alpha = 1, int gamma = 2)
        {
            var focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma, name: "Pow");
            return focal_loss;
        }

        public Tensor bbox_giou(Tensor boxes1, Tensor boxes2)
        {
            boxes1 = tf.concat(new[] 
            {   
                boxes1["...", ":2"] - boxes1["...", "2:"] * 0.5f,
                boxes1["...", ":2"] + boxes1["...", "2:"] * 0.5f
            }, axis: -1);

            boxes2 = tf.concat(new[] 
            { 
                boxes2["...", ":2"] - boxes2["...", "2:"] * 0.5f,
                boxes2["...", ":2"] + boxes2["...", "2:"] * 0.5f
            }, axis: -1);

            boxes1 = tf.concat(new[] 
            { 
                tf.minimum(boxes1["...", ":2"], boxes1["...", "2:"]),
                tf.maximum(boxes1["...", ":2"], boxes1["...", "2:"])
            }, axis: -1);

            boxes2 = tf.concat(new[] 
            { 
                tf.minimum(boxes2["...", ":2"], boxes2["...", "2:"]),
                tf.maximum(boxes2["...", ":2"], boxes2["...", "2:"])
            }, axis: -1);

            var boxes1_area = (boxes1["...", "2"] - boxes1["...", "0"]) * (boxes1["...", "3"] - boxes1["...", "1"]);
            var boxes2_area = (boxes2["...", "2"] - boxes2["...", "0"]) * (boxes2["...", "3"] - boxes2["...", "1"]);

            var left_up = tf.maximum(boxes1["...", ":2"], boxes2["...", ":2"]);
            var right_down = tf.minimum(boxes1["...", "2:"], boxes2["...", "2:"]);

            var inter_section = tf.maximum(right_down - left_up, 0.0f);
            var inter_area = inter_section["...", "0"] * inter_section["...", "1"];
            var union_area = boxes1_area + boxes2_area - inter_area;
            var iou = inter_area / union_area;

            var enclose_left_up = tf.minimum(boxes1["...", ":2"], boxes2["...", ":2"]);
            var enclose_right_down = tf.maximum(boxes1["...", "2:"], boxes2["...", "2:"]);
            var enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0f);
            var enclose_area = enclose["...", "0"] * enclose["...", "1"];
            var giou = iou - 1.0f * (enclose_area - union_area) / enclose_area;

            return giou;
        }

        public Tensor bbox_iou(Tensor boxes1, Tensor boxes2)
        {
            var boxes1_area = boxes1["...", "2"] * boxes1["...", "3"];
            var boxes2_area = boxes2["...", "2"] * boxes2["...", "3"];

            boxes1 = tf.concat(new[]
            {
                boxes1["...", ":2"] - boxes1["...", "2:"] * 0.5,
                boxes1["...", ":2"] + boxes1["...", "2:"] * 0.5
            }, axis: -1);
            boxes2 = tf.concat(new[] 
            { 
                boxes2["...", ":2"] - boxes2["...", "2:"] * 0.5,
                boxes2["...", ":2"] + boxes2["...", "2:"] * 0.5
            }, axis: -1);

            var left_up = tf.maximum(boxes1["...", ":2"], boxes2["...", ":2"]);
            var right_down = tf.minimum(boxes1["...", "2:"], boxes2["...", "2:"]);

            var inter_section = tf.maximum(right_down - left_up, 0.0f);
            var inter_area = inter_section["...", "0"] * inter_section["...", "1"];
            var union_area = boxes1_area + boxes2_area - inter_area;
            var iou = 1.0f * inter_area / union_area;

            return iou;
        }

        public void Train(TrainingOptions options)
        {
            // tf.debugging.set_log_device_placement(true);
            // tf.Context.Config.GpuOptions.AllowGrowth = true;

            var input_layer = keras.layers.Input(cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.BATCH_SIZE);
            var conv_tensors = yolo.Apply(input_layer);

            var output_tensors = new Tensors();
            foreach (var (i, conv_tensor) in enumerate(conv_tensors))
            {
                var pred_tensor = yolo.Decode(conv_tensor, i);
                output_tensors.Add(conv_tensor);
                output_tensors.Add(pred_tensor);
            }

            Model model = keras.Model(input_layer, output_tensors);
            model.summary();

            // download wights from https://drive.google.com/file/d/1J5N5Pqf1BG1sN_GWDzgViBcdK2757-tS/view?usp=sharing
            model.load_weights("./YOLOv3/yolov3.h5");

            optimizer = keras.optimizers.Adam();
            _trainingOptions = options as YoloTrainingOptions;
            int steps_per_epoch = _trainingOptions.TrainingData.Length;
            total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch;
            warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch;

            float loss = 1000;
            foreach (var epoch in range(cfg.TRAIN.EPOCHS))
            {
                print($"EPOCH {epoch + 1:D4}/{cfg.TRAIN.EPOCHS:D4}");
                float current_loss = -1;
                var watch = new Stopwatch();
                foreach (var dataset in _trainingOptions.TrainingData)
                {
                    watch.Restart();
                    current_loss = TrainStep(model, dataset.Image, dataset.Targets).numpy();
                    print($"Spent {watch.ElapsedMilliseconds} ms.");
                }

                if (current_loss < loss)
                {
                    loss = current_loss;
                    model.save_weights($"./YOLOv3/yolov3.{loss:F2}.h5");
                }
            }
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
            foreach (var (i, target) in enumerate(targets))
            {
                var (conv, pred) = (pred_result[i * 2], pred_result[i * 2 + 1]);
                var loss_items = yolo.compute_loss(pred, conv, target.Label, target.BorderBox, i);
                giou_loss += loss_items[0];
                conf_loss += loss_items[1];
                prob_loss += loss_items[2];
            }

            var total_loss = giou_loss + conf_loss + prob_loss;

            GC.Collect();
            GC.WaitForPendingFinalizers();

            var gradients = tape.gradient(total_loss, model.TrainableVariables);
            optimizer.apply_gradients(zip(gradients, model.TrainableVariables.Select(x => x as ResourceVariable)));

            float lr = optimizer.lr.numpy();
            print($"=> STEP {global_steps:D4}/{total_steps:D4} lr:{lr} giou_loss: {giou_loss.numpy()} conf_loss: {conf_loss.numpy()} prob_loss: {prob_loss.numpy()} total_loss: {total_loss.numpy()}");
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
            optimizer.lr.assign(lr);

            return total_loss;
        }

        public void SetModelArgs<T>(T args)
        {
            cfg = args as YoloConfig;
            yolo = new YOLOv3(cfg);
        }

        public ModelPredictResult Predict(Tensor input)
        {
            throw new System.NotImplementedException();
        }

        public void Config(TaskOptions options)
        {
            
        }
    }
}
