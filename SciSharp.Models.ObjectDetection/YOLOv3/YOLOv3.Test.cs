using System.Linq;
using System.Collections.Generic;
using Tensorflow;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using System.IO;
using Tensorflow.NumPy;
using Tensorflow.Keras.Engine;
using SharpCV;
using static SharpCV.Binding;

namespace SciSharp.Models.ObjectDetection
{
    public partial class YOLOv3
    {
        public ModelTestResult Test(TestingOptions options)
        {
            var input_layer = keras.layers.Input(cfg.TEST.INPUT_SIZE);
            var feature_maps = yolo.Apply(input_layer);

            var tensors = new List<Tensor>();
            foreach (var (i, fm) in enumerate(feature_maps))
            {
                var bbox_tensor = yolo.Decode(fm, i);
                tensors.Add(bbox_tensor);
            }
            var bbox_tensors = new Tensors(tensors);
            var model = tf.keras.Model(input_layer, bbox_tensors);

            model.load_weights("./YOLOv3/yolov3.h5");

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
                pred_bbox = pred_bbox.Select(x => tf.reshape(x, (-1, x.shape[-1]))).ToList();
                var pred_bbox_concat = tf.concat(pred_bbox, axis: 0);
                var bboxes = Utils.postprocess_boxes(pred_bbox_concat.numpy(), image_size, cfg.TEST.INPUT_SIZE[0], cfg.TEST.SCORE_THRESHOLD);
                if (bboxes.size > 0)
                {
                    var best_box_results = Utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method: "nms");
                    Utils.draw_bbox(image, best_box_results, yolo.Classes.Values.ToArray());
                    cv2.imwrite(Path.Combine(cfg.TEST.DECTECTED_IMAGE_PATH, Path.GetFileName(image_name)), image);
                }
            }

            return new ModelTestResult();
        }
    }
}
