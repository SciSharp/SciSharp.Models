using NumSharp;
using SharpCV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using static SharpCV.Binding;
using static Tensorflow.Binding;

namespace SciSharp.Models.YOLOv3
{
    public class Utils
    {
        public static Dictionary<int, string> read_class_names(string file)
        {
            var classes = new Dictionary<int, string>();
            foreach (var line in File.ReadAllLines(file))
                classes[classes.Count] = line;
            return classes;
        }

        public static NDArray get_anchors(string file)
        {
            return np.array(File.ReadAllText(file).Split(',')
                .Select(x => float.Parse(x))
                .ToArray()).reshape(3, 3, 2);
        }

        public static (NDArray, NDArray) image_preporcess(Mat image, int[] target_size, NDArray gt_boxes = null)
        {
            var dst = cv2.cvtColor(image, ColorConversionCodes.COLOR_BGR2RGB);

            var (ih, iw) = (target_size[0], target_size[1]);
            var (h, w) = (image.shape[0], image.shape[1]);

            var scale = Math.Min(iw / (w + 0f), ih / (h + 0f));
            var (nw, nh) = (Convert.ToInt32(scale * w), Convert.ToInt32(scale * h));

            NDArray image_resized = cv2.resize(dst, (nw, nh));

            var image_paded = np.full((ih, iw, 3), fill_value: 128.0f);
            var (dw, dh) = ((iw - nw) % 2, (ih - nh) % 2);

            image_paded[new Slice(dh, nh + dh), new Slice(dw, nw + dw), Slice.All] = image_resized;
            image_paded = image_paded / 255;

            if (gt_boxes == null)
            {
                return (image_paded, gt_boxes);
            }
            else
            {
                gt_boxes[Slice.All, new Slice(0, 4, 2)] = gt_boxes[Slice.All, new Slice(0, 4, 2)] * scale + dw;
                gt_boxes[Slice.All, new Slice(1, 4, 2)] = gt_boxes[Slice.All, new Slice(1, 4, 2)] * scale + dh;

                /*foreach (var nd in gt_boxes.GetNDArrays())
                {
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(0) * scale) + dw, 0);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(2) * scale) + dw, 2);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(1) * scale) + dh, 1);
                    nd.SetInt32(Convert.ToInt32(nd.GetInt32(3) * scale) + dh, 3);
                }*/
                
                return (image_paded, gt_boxes);
            }
        }

        public static NDArray postprocess_boxes(NDArray pred_bbox, TensorShape org_img_shape, int input_size, float score_threshold)
        {
            var valid_scale = np.array(0, np.inf).astype(np.float32);
            var pred_xywh = pred_bbox[Slice.All, new Slice(0, 4)];
            var pred_conf = pred_bbox[Slice.All, 4];
            var pred_prob = pred_bbox[Slice.All, new Slice(5)];

            // (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
            var pred_coor = np.concatenate(new[]
            {
                pred_xywh[Slice.All, new Slice(0, 2)] - pred_xywh[Slice.All, new Slice(2)] * 0.5f,
                pred_xywh[Slice.All, new Slice(0, 2)] + pred_xywh[Slice.All, new Slice(2)] * 0.5f
            }, axis: -1);

            // (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
            var (org_h, org_w) = org_img_shape;
            var resize_ratio = min(input_size / org_w, input_size / org_h);
            var dw = (input_size - resize_ratio * org_w) / 2;
            var dh = (input_size - resize_ratio * org_h) / 2;

            pred_coor[Slice.All, new Slice(0, 4, 2)] = 1.0 * (pred_coor[Slice.All, new Slice(0, 4, 2)] - dw) / resize_ratio;
            pred_coor[Slice.All, new Slice(1, 4, 2)] = 1.0 * (pred_coor[Slice.All, new Slice(1, 4, 2)] - dh) / resize_ratio;

            // (3) clip some boxes those are out of range
            pred_coor = np.concatenate(new[] 
            {
                np.maximum(pred_coor[Slice.All, new Slice(0, 2)], np.array(0, 0)),
                np.minimum(pred_coor[Slice.All, new Slice(2)], np.array(org_w - 1, org_h - 1))
            }, axis: -1);
            var invalid_mask = np.logical_or(pred_coor[Slice.All, 0] > pred_coor[Slice.All, 2], pred_coor[Slice.All, 1] > pred_coor[Slice.All, 3]);
            pred_coor[invalid_mask] = 0;

            // (4) discard some invalid boxes
            var xx = pred_coor[Slice.All, new Slice(2, 4)] - pred_coor[Slice.All, new Slice(0, 2)];
            var bboxes_scale = np.sqrt(np.multiply(xx[Slice.All, 0], xx[Slice.All, 1]));
            var scale_mask = np.logical_and(valid_scale[0] < bboxes_scale, bboxes_scale < valid_scale[1]);

            // (5) discard some boxes with low scores
            var classes = np.argmax(pred_prob, axis: -1);
            var scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes];
            var score_mask = scores > score_threshold;
            var mask = np.logical_and(scale_mask, score_mask);
            NDArray coors;
            (coors, scores, classes) = (pred_coor[mask], scores[mask], classes[mask]);

            if (coors.size == 0)
                return coors;

            return np.concatenate(new[]
            {
                coors,
                scores[Slice.All, np.newaxis],
                classes[Slice.All, np.newaxis]
            }, axis: -1);
        }

        public static Mat draw_bbox(Mat image, List<NDArray> bboxes, string[] classes, bool show_label = true)
        {
            var num_classes = len(classes);
            var (image_h, image_w, _) = image.shape;
            var hsv_tuples = range(num_classes).Select(x => (1.0f * x / num_classes, 1f, 1f)).ToArray();
            var colors = hsv_tuples.Select(x => 
            (
                Convert.ToInt32(x.Item1 * 255), 
                Convert.ToInt32(x.Item2 * 255), 
                Convert.ToInt32(x.Item3 * 255))
            ).ToArray();

            foreach (var (i, bbox) in enumerate(bboxes))
            {
                var coor = bbox[new Slice(0, 4)].astype(np.int32);
                var fontScale = 0.5f;
                float score = bbox[4];
                int class_ind = bbox[5].astype(np.int32);
                var bbox_color = colors[class_ind];
                var bbox_thick = Convert.ToInt32(Math.Floor(0.6f * (image_h + image_w) / 600));
                var (c1, c2) = (((int)coor[0], (int)coor[1]), ((int)coor[2], (int)coor[3]));
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick);

                if (show_label)
                {
                    var bbox_mess = $"{classes[class_ind]}: {score:F2}";
                    var t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness: bbox_thick % 2);
                    cv2.rectangle(image, c1, (c1.Item1 + t_size.Width, c1.Item2 - t_size.Height - 3), bbox_color, -1);
                    cv2.putText(image, bbox_mess, (c1.Item1, c1.Item2 - 2),
                        HersheyFonts.HERSHEY_SIMPLEX, fontScale, (0, 0, 0),
                        thickness: bbox_thick % 2,
                        lineType: LineTypes.LINE_AA);
                }
            }

            return image;
        }

        public static NDArray bboxes_iou(NDArray boxes1, NDArray boxes2)
        {
            var boxes1_area = (boxes1[Slice.Ellipsis, 2] - boxes1[Slice.Ellipsis, 0]) * (boxes1[Slice.Ellipsis, 3] - boxes1[Slice.Ellipsis, 1]);
            var boxes2_area = (boxes2[Slice.Ellipsis, 2] - boxes2[Slice.Ellipsis, 0]) * (boxes2[Slice.Ellipsis, 3] - boxes2[Slice.Ellipsis, 1]);

            var left_up = np.maximum(boxes1[Slice.Ellipsis, new Slice(0, 2)], boxes2[Slice.Ellipsis, new Slice(0, 2)]);
            var right_down = np.minimum(boxes1[Slice.Ellipsis, new Slice(2)], boxes2[Slice.Ellipsis, new Slice(2)]);

            var inter_section = np.maximum(right_down - left_up, 0.0f);
            var inter_area = inter_section[Slice.Ellipsis, 0] * inter_section[Slice.Ellipsis, 1];
            var union_area = boxes1_area + boxes2_area - inter_area;
            var ious = np.maximum(1.0f * inter_area / union_area, 1.1920929e-07f);

            return ious;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="bboxes">(xmin, ymin, xmax, ymax, score, class)</param>
        /// <param name="iou_threshold"></param>
        /// <param name="sigma"></param>
        /// <param name="method"></param>
        /// <returns></returns>
        public static List<NDArray> nms(NDArray bboxes, float iou_threshold, float sigma= 0.3f, string method = "nms")
        {
            var classes_in_img = bboxes[Slice.All, 5].Data<float>()
                .Distinct()
                .OrderBy(x => x)
                .ToArray();
            var best_bboxes = new List<NDArray>();
            foreach (float cls in classes_in_img)
            {
                var cls_mask = bboxes[Slice.All, 5] == cls;
                var cls_bboxes = bboxes[cls_mask];
                while(len(cls_bboxes) > 0)
                {
                    var max_ind = np.argmax(cls_bboxes[Slice.All, 4]);
                    var best_bbox = cls_bboxes[max_ind];
                    best_bboxes.Add(best_bbox);
                    cls_bboxes = np.concatenate(new[] { cls_bboxes[new Slice(0, max_ind)], cls_bboxes[new Slice(max_ind + 1)] });
                    if (len(cls_bboxes) == 0)
                        continue;
                    var iou = bboxes_iou(best_bbox[np.newaxis, new Slice(0, 4)], cls_bboxes[Slice.All, new Slice(0, 4)]);
                    var weight = np.ones(new Shape(len(iou)), np.float32);

                    if(method == "nms")
                    {
                        var iou_mask = iou > iou_threshold;
                        weight[iou_mask] = 0.0f;
                    }

                    cls_bboxes[Slice.All, 4] = cls_bboxes[Slice.All, 4] * weight;
                    var score_mask = cls_bboxes[Slice.All, 4] > 0f;
                    cls_bboxes = cls_bboxes[score_mask];
                }
            }
            return best_bboxes;
        }
    }
}
