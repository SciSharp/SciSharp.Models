using SciSharp.Models.Exceptions;
using System.IO;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace SciSharp.Models.ImageClassification
{
    public partial class TransferLearning 
    {
        Session predictSession;
        public ModelPredictResult Predict(string imagePath)
        {
            if (!File.Exists(options.ModelPath))
                throw new FreezedGraphNotFoundException();

            if (labels == null)
                labels = File.ReadAllLines(options.LabelPath);

            var image = ReadTensorFromImageFile(imagePath);
            readImageSession.graph.Exit();

            // import graph and variables
            if (predictSession == null)
            {
                var graph = tf.Graph();
                graph.Import(options.ModelPath);
                predictSession = tf.Session(graph);
            }
            else
            {
                predictSession.graph.as_default();
            }

            Tensor input = predictSession.graph.OperationByName(input_tensor_name);
            Tensor output = predictSession.graph.OperationByName(final_tensor_name);
            var result = predictSession.run(output, (input, image));
            predictSession.graph.Exit();
            var prob = np.squeeze(result);
            var idx = np.argmax(prob);

            print($"{Path.GetFileName(imagePath)}, {labels[idx]}, {(float)prob[idx]}");

            return new ModelPredictResult
            {
                Label = labels[idx],
                Probability = prob[idx]
            };
        }

        Session readImageSession;
        NDArray ReadTensorFromImageFile(string file_name,
            int input_height = 299,
            int input_width = 299,
            int input_mean = 0,
            int input_std = 255)
        {
            if (readImageSession == null)
            {
                var graph = tf.Graph().as_default();
                var file_reader = tf.io.read_file(file_name, "file_reader");
                var image_reader = tf.image.decode_jpeg(file_reader, channels: 3, name: "jpeg_reader");
                var caster = tf.cast(image_reader, tf.float32);
                var dims_expander = tf.expand_dims(caster, 0);
                var resize = tf.constant(new int[] { input_height, input_width });
                var bilinear = tf.image.resize_bilinear(dims_expander, resize);
                var sub = tf.subtract(bilinear, new float[] { input_mean });
                var normalized = tf.divide(sub, new float[] { input_std });
                readImageSession = tf.Session(graph);
                return readImageSession.run(normalized);
            }
            else
            {
                readImageSession.graph.as_default();
                Tensor normalized = readImageSession.graph.OperationByName("truediv");
                return readImageSession.run(normalized);
            }
        }
    }
}
