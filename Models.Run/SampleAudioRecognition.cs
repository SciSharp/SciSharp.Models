using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Models.Run
{
    /// <summary>
    /// https://www.tensorflow.org/tutorials/audio/simple_audio
    /// </summary>
    public class SampleAudioRecognition
    {
        public void Run()
        {
            // Set seed for experiment reproducibility
            int seed = 42;
            tf.set_random_seed(seed);
            var data_dir = Path.Combine("data", "mini_speech_commands");
            Directory.CreateDirectory(data_dir);
            keras.utils.get_file("mini_speech_commands.zip",
                origin: "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract: true,
                cache_dir: ".",
                cache_subdir: "data");

            var commands = np.array(tf.io.gfile.listdir(data_dir));
            var files = tf.io.gfile.glob(data_dir);
            var filenames = tf.random_shuffle(tf.constant(files));

            // Split the files into training, validation and test sets using a 80:10:10 ratio, respectively.
            var train_files = filenames[":6400"];
            var val_files = filenames[$"6400:{6400 + 800}"];
            var test_files = filenames["-800:"];

            //{
            //    var file_path = filenames[0];
            //    var parts = tf.strings.split(file_path, sep: Path.DirectorySeparatorChar);
            //    var part = parts[0][-2];
            //    var audio_binary = tf.io.read_file(file_path);
            //    var (waveform, _) = tf.audio.decode_wav(audio_binary);
            //}

            var files_ds = tf.data.Dataset.from_tensor_slices(train_files);
            var waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls: tf.data.AUTOTUNE);

            foreach (var (waveform, label) in waveform_ds.take(1))
            {
                print(label);
                var spectrogram = get_spectrogram(waveform);
            }

            var spectrogram_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls: tf.data.AUTOTUNE);
        }

        public void LoadModel(string modelPath)
        {
            keras.models.load_model(modelPath);
        }

        Tensors get_waveform_and_label(Tensors file_path)
        {
            var parts = tf.strings.split(file_path, sep: Path.DirectorySeparatorChar);
            var label = parts[0][-2];
            var audio_binary = tf.io.read_file(file_path);
            var (audio, _) = tf.audio.decode_wav(audio_binary);
            var waveform = tf.squeeze(audio, axis: -1);
            return (waveform, label);
        }

        Tensors get_spectrogram_and_label_id(Tensors inputs)
        {
            var (audio, label) = inputs;
            var spectrogram = get_spectrogram(audio);
            return spectrogram;
        }

        Tensor get_spectrogram(Tensor waveform)
        {
            var zero_padding = tf.zeros(tf.constant(new[] { 16000 }) - tf.shape(waveform), dtype: tf.float32);
            return zero_padding;
        }
    }
}
