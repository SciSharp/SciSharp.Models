using NumSharp;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.IO;

namespace SciSharp.Models.YOLOv3
{
    public class YOLOv3
    {
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

            var output_tensors = new List<Tensor>();

            var pred_tensor = Decode(conv_sbbox, 0);
            output_tensors.Add(conv_sbbox);
            output_tensors.Add(pred_tensor);

            pred_tensor = Decode(conv_mbbox, 1);
            output_tensors.Add(conv_mbbox);
            output_tensors.Add(pred_tensor);

            pred_tensor = Decode(conv_lbbox, 2);
            output_tensors.Add(conv_lbbox);
            output_tensors.Add(pred_tensor);

            return output_tensors.ToArray();
        }

        public Tensor Decode(Tensor conv_output, int i = 0)
        {
            var conv_shape = conv_output.shape; // tf.shape(conv_output);
            var batch_size = 4; // conv_shape[0];
            var output_size = conv_shape[1];

            conv_output = keras.layers.Reshape((output_size, output_size, 3, 5 + num_class)).Apply(conv_output);

            var conv_raw_dxdy = conv_output[":", ":", ":", ":", "0:2"];
            var conv_raw_dwdh = conv_output[":", ":", ":", ":", "2:4"];
            var conv_raw_conf = conv_output[":", ":", ":", ":", "4:5"];
            var conv_raw_prob = conv_output[":", ":", ":", ":", "5:"];

            var range = tf.range(output_size, dtype: tf.int32);
            var y = tf.tile(range[Slice.All, tf.newaxis], new[] { 1, output_size });
            var x = tf.tile(range[tf.newaxis, Slice.All], new[] { output_size, 1 });

            Tensor xy_grid = keras.layers.Concatenate(axis: -1).Apply(new[] { x[Slice.All, Slice.All, tf.newaxis], y[Slice.All, Slice.All, tf.newaxis] });
            xy_grid = tf.tile(xy_grid[tf.newaxis, Slice.All, Slice.All, tf.newaxis, Slice.All], new[] { batch_size, 1, 1, 3, 1 });
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
            var conv_shape = tf.shape(conv).ToArray<int>();
            var batch_size = conv_shape[0];
            var output_size = conv_shape[1];
            var input_size = tf.constant(strides[i] * output_size);
            conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_class));

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
            boxes1 = tf.concat(new[] { boxes1["...", ":2"] - boxes1["...", "2:"] * 0.5f,
                            boxes1["...", ":2"] + boxes1["...", "2:"] * 0.5f}, axis: -1);
            boxes2 = tf.concat(new[] { boxes2["...", ":2"] - boxes2["...", "2:"] * 0.5f,
                            boxes2["...", ":2"] + boxes2["...", "2:"] * 0.5f}, axis: -1);

            boxes1 = tf.concat(new[] { tf.minimum(boxes1["...", ":2"], boxes1["...", "2:"]),
                            tf.maximum(boxes1["...", ":2"], boxes1["...", "2:"])}, axis: -1);
            boxes2 = tf.concat(new[] { tf.minimum(boxes2["...", ":2"], boxes2["...", "2:"]),
                            tf.maximum(boxes2["...", ":2"], boxes2["...", "2:"])}, axis: -1);

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

            boxes1 = tf.concat(new[] { boxes1["...", ":2"] - boxes1["...", "2:"] * 0.5,
                            boxes1["...", ":2"] + boxes1["...", "2:"] * 0.5}, axis: -1);
            boxes2 = tf.concat(new[] { boxes2["...", ":2"] - boxes2["...", "2:"] * 0.5,
                            boxes2["...", ":2"] + boxes2["...", "2:"] * 0.5}, axis: -1);

            var left_up = tf.maximum(boxes1["...", ":2"], boxes2["...", ":2"]);
            var right_down = tf.minimum(boxes1["...", "2:"], boxes2["...", "2:"]);

            var inter_section = tf.maximum(right_down - left_up, 0.0f);
            var inter_area = inter_section["...", "0"] * inter_section["...", "1"];
            var union_area = boxes1_area + boxes2_area - inter_area;
            var iou = 1.0f * inter_area / union_area;

            return iou;
        }
    }
}
