using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace SciSharp.Models
{
    public class ImageUtil
    {
        public static Tensor ReadImageFromFile(string file_name,
            int input_height = 299,
            int input_width = 299,
            int channels = 3,
            int input_mean = 0,
            int input_std = 255)
        {
            tf.enable_eager_execution();
            var file_reader = tf.io.read_file(file_name, "file_reader");
            var image_reader = tf.image.decode_jpeg(file_reader, channels: channels, name: "jpeg_reader");
            var caster = tf.cast(image_reader, tf.float32);
            var dims_expander = tf.expand_dims(caster, 0);
            var resize = tf.constant(new int[] { input_height, input_width });
            var bilinear = tf.image.resize_bilinear(dims_expander, resize);
            var sub = tf.subtract(bilinear, new float[] { input_mean });
            var normalized = tf.divide(sub, new float[] { input_std });
            tf.Context.restore_mode();
            return normalized;
        }
    }
}
