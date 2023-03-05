using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using Tensorflow;
using static Tensorflow.Binding;

namespace SciSharp.Models.ImageClassification
{
    public partial class CNN
    {
        void SaveCheckpoint(Session sess)
        {
            var checkpoint = Path.Combine(_taskDir, "checkpoint", "checkpoint.ckpt");
            print($"Saving checkpoint to {checkpoint} ...");
            var saver = tf.train.Saver();
            saver.save(sess, checkpoint);
        }

        void RestoreCheckpoint()
        {
            var graph = tf.Graph().as_default();
            var sess = tf.Session(graph);
            var saver = tf.train.import_meta_graph(Path.Combine(_taskDir, "mnist_cnn.ckpt.meta"));
            // Restore variables from checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(_taskDir));

            var loss = graph.get_tensor_by_name("Train/Loss/loss:0");
            var accuracy = graph.get_tensor_by_name("Train/Accuracy/accuracy:0");
            var x = graph.get_tensor_by_name("Input/X:0");
            var y = graph.get_tensor_by_name("Input/Y:0");

            //var init = tf.global_variables_initializer();
            //sess.run(init);
        }
    }
}
