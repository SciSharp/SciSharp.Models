using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;
using static SharpCV.Binding;
using SharpCV;
using Tensorflow.NumPy;

namespace SciSharp.Models.ObjectDetection
{
    public class YoloDataset : IEnumerable<BatchFeedingImage>
    {
        string annot_path;
        int[] input_sizes;
        int batch_size;
        bool data_aug;
        int train_input_size;
        int[] train_input_sizes;
        NDArray train_output_sizes;
        NDArray strides;
        NDArray anchors;
        Dictionary<int, string> classes;
        int num_classes;
        int anchor_per_scale;
        int max_bbox_per_scale;
        string[] annotations;
        int num_samples;
        int num_batchs;
        int batch_count;

        public int Length => num_batchs;

        public YoloDataset(string dataset_type, YoloConfig cfg)
        {
            annot_path = dataset_type == "train" ? cfg.TRAIN.ANNOT_PATH : cfg.TEST.ANNOT_PATH;
            input_sizes = dataset_type == "train" ? cfg.TRAIN.INPUT_SIZE : cfg.TEST.INPUT_SIZE;
            batch_size = dataset_type == "train" ? cfg.TRAIN.BATCH_SIZE : cfg.TEST.BATCH_SIZE;
            data_aug = dataset_type == "train" ? cfg.TRAIN.DATA_AUG : cfg.TEST.DATA_AUG;
            train_input_sizes = cfg.TRAIN.INPUT_SIZE;
            strides = np.array(cfg.YOLO.STRIDES);

            classes = Utils.read_class_names(cfg.YOLO.CLASSES);
            num_classes = classes.Count;
            anchors = Utils.get_anchors(cfg.YOLO.ANCHORS);
            anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE;
            max_bbox_per_scale = 150;

            annotations = load_annotations();
            num_samples = len(annotations);
            num_batchs = Convert.ToInt32(Math.Ceiling(num_samples / Convert.ToDecimal(batch_size)));
            batch_count = 0;
        }

        string[] load_annotations()
        {
            var annotations = File.ReadAllLines(annot_path);
            var shuffled = tf.random_shuffle(tf.constant(annotations, dtype: tf.@string));
            return shuffled.StringData();
        }

        private (NDArray, NDArray) parse_annotation(string annotation)
        {
            var line = annotation.Split();
            var image_path = line[0];
            if (!File.Exists(image_path))
                throw new KeyError($"{image_path} does not exist ... ");
            var image = cv2.imread(image_path);

            var bboxes = np.stack(line
                .Skip(1)
                .Select(box => np.array(box
                        .Split(',')
                        .Select(x => float.Parse(x))
                        .ToArray()))
                .ToArray());
            
            if (data_aug)
            {
                (image, bboxes) = random_horizontal_flip(image, bboxes);
                (image, bboxes) = random_crop(image, bboxes);
                (image, bboxes) = random_translate(image, bboxes);
            }
            image = cv2.cvtColor(image, ColorConversionCodes.COLOR_BGR2RGB);
            var (image1, bboxes1) = Utils.image_preporcess(image, new[] { train_input_size, train_input_size }, bboxes);
            // cv2.imshow("ss", image);
            // cv2.waitKey(0);
            return (image1, bboxes1);
        }

        private(NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) preprocess_true_boxes(NDArray bboxes)
        {
            var labels = range(3).Select(i => np.zeros((train_output_sizes[i], train_output_sizes[i], anchor_per_scale, 5 + num_classes), dtype: np.float32)).ToArray();
            var bboxes_xywh = range(3).Select(x => np.zeros((max_bbox_per_scale, 4), dtype: np.float32)).ToArray();
            var bbox_count = np.zeros(new Shape(3), np.int32);

            foreach(var bbox in bboxes)
            {
                var bbox_coor = bbox[":4"];
                int bbox_class_ind = bbox[4];

                var onehot = np.zeros(new Shape(num_classes), dtype: np.float32);
                onehot[bbox_class_ind] = 1.0f;
                var uniform_distribution = np.full(new Shape(num_classes), 1.0f / num_classes);
                var deta = 0.01f;
                var smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution;

                var bbox_xywh = np.concatenate(new[]
                {
                    (bbox_coor["2:"] + bbox_coor[":2"]) * 0.5f,
                    (bbox_coor["2:"] - bbox_coor[":2"]) * 1.0f
                }, axis: -1);
                var bbox_xywh_scaled = 1.0f * bbox_xywh[np.newaxis, Slice.All] / strides[Slice.All, np.newaxis];
                var iou = new List<NDArray>();
                var exist_positive = false;
                foreach(var i in range(3))
                {
                    var anchors_xywh = np.zeros((anchor_per_scale, 4), dtype: np.float32);
                    anchors_xywh[Slice.All, new Slice(0, 2)] = np.floor(bbox_xywh_scaled[i, new Slice(0, 2)]) + 0.5f;
                    anchors_xywh[Slice.All, new Slice(2, 4)] = anchors[i];

                    var iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, Slice.All], anchors_xywh);
                    iou.Add(iou_scale);
                    var iou_mask = iou_scale > 0.3f;
                    if (np.any(iou_mask))
                    {
                        var floors = np.floor(bbox_xywh_scaled[i, new Slice(0, 2)]).astype(np.int32);
                        (int xind, int yind) = (floors[0], floors[1]);

                        // set value by mask
                        foreach(var (mask_index, is_mask) in enumerate(iou_mask.ToArray<bool>()))
                        {
                            if (!is_mask) 
                                continue;
                            var label = labels[i];
                            label[yind, xind, mask_index, new Slice(0, 4)] = bbox_xywh;
                            label[yind, xind, mask_index, new Slice(4, 5)] = 1.0f;
                            label[yind, xind, mask_index, new Slice(5)] = smooth_onehot;
                        }

                        int bbox_ind = bbox_count[i] % max_bbox_per_scale;
                        bboxes_xywh[i][bbox_ind, new Slice(0, 4)] = bbox_xywh;
                        bbox_count[i] += 1;
                        exist_positive = true;
                    }
                }

                if (!exist_positive)
                {
                    throw new NotImplementedException("");
                }
            }

            var (label_sbbox, label_mbbox, label_lbbox) = (labels[0], labels[1], labels[2]);
            var (sbboxes, mbboxes, lbboxes) = (bboxes_xywh[0], bboxes_xywh[1], bboxes_xywh[2]);

            return (label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes);
        }

        private NDArray bbox_iou(NDArray boxes1, NDArray boxes2)
        {
            var boxes1_area = boxes1[Slice.Ellipsis, 2] * boxes1[Slice.Ellipsis, 3];
            var boxes2_area = boxes2[Slice.Ellipsis, 2] * boxes2[Slice.Ellipsis, 3];

            boxes1 = np.concatenate(new[]
            {
                boxes1[Slice.Ellipsis, new Slice(":2")] - boxes1[Slice.Ellipsis, new Slice("2:")] * 0.5f,
                boxes1[Slice.Ellipsis, new Slice(":2")] + boxes1[Slice.Ellipsis, new Slice("2:")] * 0.5f
            }, axis: -1);

            boxes2 = np.concatenate(new[]
            {
                boxes2[Slice.Ellipsis, new Slice(":2")] - boxes2[Slice.Ellipsis, new Slice("2:")] * 0.5f,
                boxes2[Slice.Ellipsis, new Slice(":2")] + boxes2[Slice.Ellipsis, new Slice("2:")] * 0.5f
            }, axis: -1);

            var left_up = np.maximum(boxes1[Slice.Ellipsis, new Slice(":2")], boxes2[Slice.Ellipsis, new Slice(":2")]);
            var right_down = np.minimum(boxes1[Slice.Ellipsis, new Slice("2:")], boxes2[Slice.Ellipsis, new Slice("2:")]);
            var inter_section = np.maximum(right_down - left_up, NDArray.Scalar(0.0f));
            var inter_area = inter_section[Slice.Ellipsis, 0] * inter_section[Slice.Ellipsis, 1];
            var union_area = boxes1_area + boxes2_area - inter_area;

            return inter_area / union_area;
        }

        private (Mat, NDArray) random_horizontal_flip(Mat image, NDArray bboxes)
        {
            var rand = new Random();
            if (rand.NextDouble() < 0.5)
            {
                /*var w = image.shape[1];
                image = cv2.flip(image, FLIP_MODE.FLIP_HORIZONTAL_MODE);
                (bboxes[":", 0], bboxes[":", 2]) = (w - bboxes[":", 2], w - bboxes[":", 0]);*/
            }

            return (image, bboxes);
        }

        private (Mat, NDArray) random_crop(Mat image, NDArray bboxes)
        {
            return (image, bboxes);
        }

        private (Mat, NDArray) random_translate(Mat image, NDArray bboxes)
        {
            return (image, bboxes);
        }

        public IEnumerator<BatchFeedingImage> GetEnumerator()
        {
            tf.device("/cpu:0");

            train_input_size = train_input_sizes[new Random().Next(0, train_input_sizes.Length - 1)];
            train_output_sizes = train_input_size / strides;
            var batch_image = np.zeros((batch_size, train_input_size, train_input_size, 3), np.float32);
            var batch_label_sbbox = np.zeros((batch_size, train_output_sizes[0], train_output_sizes[0],
                                          anchor_per_scale, 5 + num_classes), np.float32);
            var batch_label_mbbox = np.zeros((batch_size, train_output_sizes[1], train_output_sizes[1],
                                          anchor_per_scale, 5 + num_classes), np.float32);
            var batch_label_lbbox = np.zeros((batch_size, train_output_sizes[2], train_output_sizes[2],
                                          anchor_per_scale, 5 + num_classes), np.float32);

            var batch_sbboxes = np.zeros((batch_size, max_bbox_per_scale, 4), np.float32);
            var batch_mbboxes = np.zeros((batch_size, max_bbox_per_scale, 4), np.float32);
            var batch_lbboxes = np.zeros((batch_size, max_bbox_per_scale, 4), np.float32);

            int num = 0;
            while (batch_count < num_batchs)
            {
                while (num < batch_size)
                {
                    var index = batch_count * batch_size + num;
                    if (index >= num_samples)
                        index -= num_samples;
                    var annotation = annotations[index];
                    var (image, bboxes) = parse_annotation(annotation);
                    var (label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) = preprocess_true_boxes(bboxes);

                    batch_image[num, Slice.All, Slice.All, Slice.All] = image;
                    batch_label_sbbox[num, Slice.All, Slice.All, Slice.All, Slice.All] = label_sbbox;
                    batch_label_mbbox[num, Slice.All, Slice.All, Slice.All, Slice.All] = label_mbbox;
                    batch_label_lbbox[num, Slice.All, Slice.All, Slice.All, Slice.All] = label_lbbox;
                    batch_sbboxes[num, Slice.All, Slice.All] = sbboxes;
                    batch_mbboxes[num, Slice.All, Slice.All] = mbboxes;
                    batch_lbboxes[num, Slice.All, Slice.All] = lbboxes;
                    num += 1;
                }
                batch_count += 1;
                yield return new BatchFeedingImage
                {
                    Image = batch_image,
                    Targets = new List<LabelBorderBox>
                    {
                        new LabelBorderBox
                        {
                            Label = batch_label_sbbox,
                            BorderBox = batch_sbboxes
                        },
                            new LabelBorderBox
                        {
                            Label = batch_label_mbbox,
                            BorderBox = batch_mbboxes
                        },
                            new LabelBorderBox
                        {
                            Label = batch_label_lbbox,
                            BorderBox = batch_lbboxes
                        }
                    }
                };
            }

            batch_count = 0;
            annotations = tf.random_shuffle(tf.constant(annotations, dtype: tf.@string)).StringData();
        }

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();
    }
}
