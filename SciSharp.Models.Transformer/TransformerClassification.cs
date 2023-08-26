using System;
using System.Collections.Generic;
using System.Text;
using SciSharp.Models.Transformer;
using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace Tensorflow.Keras.Layers
{
    public class TransformerClassification : Layer
    {
        TransformerClassificationArgs args;
        ILayer embedding_layer;
        ILayer transformer_block;
        ILayer pooling;
        ILayer dropout1;
        ILayer dense;
        ILayer dropout2;
        ILayer output;

        public TransformerClassification(TransformerClassificationArgs args) : base(args)
        {
            this.args = args;
        }
        public override void build(KerasShapesWrapper input_shape)
        {
            _buildInputShape = input_shape;
            embedding_layer = new TokenAndPositionEmbedding(new TokenAndPositionEmbeddingArgs { Maxlen = args.Maxlen, VocabSize = args.VocabSize, EmbedDim = args.EmbedDim });
            transformer_block = new TransformerBlock(new TransformerBlockArgs { EmbedDim = args.EmbedDim, NumHeads = args.NumHeads, FfDim = args.FfDim });
            pooling = keras.layers.GlobalAveragePooling1D();
            dropout1 = keras.layers.Dropout(args.DropoutRate);
            dense = keras.layers.Dense(args.DenseDim, activation: "relu");
            dropout2 = keras.layers.Dropout(args.DropoutRate);
            output = keras.layers.Dense(2, activation: "softmax");
            StackLayers(embedding_layer, transformer_block, pooling, dropout1, dense, dropout2, output);
            built = true;
        }
        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var embeddings = embedding_layer.Apply(inputs, state, training, optional_args);
            var outputs = transformer_block.Apply(embeddings, state, training, optional_args);
            outputs = pooling.Apply(outputs, state, training, optional_args);
            outputs = dropout1.Apply(outputs, state, training, optional_args);
            outputs = dense.Apply(outputs, state, training, optional_args);
            outputs = dropout2.Apply(outputs, state, training, optional_args);
            outputs = output.Apply(outputs, state, training, optional_args);
            return outputs;
        }
        public static IModel Build(TransformerClassificationConfig cfg)
        {
            var inputs = keras.layers.Input(shape: new[] { cfg.DatasetCfg.maxlen });
            var transformer = new TransformerClassification(
                new TransformerClassificationArgs
                {
                    Maxlen = cfg.DatasetCfg.maxlen,
                    VocabSize = cfg.DatasetCfg.vocab_size,
                    EmbedDim = cfg.ModelCfg.embed_dim,
                    NumHeads = cfg.ModelCfg.num_heads,
                    FfDim = cfg.ModelCfg.ff_dim,
                    DropoutRate = cfg.ModelCfg.dropout_rate,
                    DenseDim = cfg.ModelCfg.dense_dim
                });
            var outputs = transformer.Apply(inputs);
            return keras.Model(inputs: inputs, outputs: outputs);
        }
        public static IModel Train(TransformerClassificationConfig? cfg = null)
        {
            cfg = cfg ?? new TransformerClassificationConfig();
            var dataloader = new IMDbDataset(cfg); //the dataset is initially downloaded at TEMP dir, e.g., C:\Users\{user name}\AppData\Local\Temp\imdb\imdb.npz
            var dataset = dataloader.GetData();
            var x_train = dataset[0];
            var y_train = dataset[1];
            var x_val = dataset[2];
            var y_val = dataset[3];
            var model = Build(cfg);
            model.summary();
            model.compile(optimizer: keras.optimizers.Adam(learning_rate: 0.01f), loss: keras.losses.SparseCategoricalCrossentropy(), metrics: new string[] { "accuracy" });
            model.fit((NDArray)x_train, (NDArray)y_train, batch_size: cfg.TrainCfg.batch_size, epochs: cfg.TrainCfg.epochs, validation_data: ((NDArray val_x, NDArray val_y))(x_val, y_val));
            return model;
        }
        public static void Save(IModel model, string path)
        {
            model.save_weights(path + @"\weights.h5");
        }
        public static IModel Load(string path, TransformerClassificationConfig? cfg = null)
        {
            cfg = cfg ?? new TransformerClassificationConfig();
            var model = Build(cfg);
            model.load_weights(path + @"\weights.h5");
            return model;
        }
        public static Tensors Predict(IModel model, Tensors inputs)
        {
            var outputs = model.predict(inputs);
            return outputs;
        }
        public static void Evaluate(IModel model)
        {
            var cfg = new TransformerClassificationConfig();
            var dataloader = new IMDbDataset(cfg); //the dataset is initially downloaded at TEMP dir, e.g., C:\Users\{user name}\AppData\Local\Temp\imdb\imdb.npz
            var dataset = dataloader.GetData();
            var x_train = dataset[0];
            var y_train = dataset[1];
            var x_val = dataset[2];
            var y_val = dataset[3];
            model.compile(optimizer: keras.optimizers.Adam(learning_rate: 0.01f), loss: keras.losses.SparseCategoricalCrossentropy(), metrics: new string[] { "accuracy" });
            var log = model.evaluate((NDArray)x_val, (NDArray)y_val);
            Console.WriteLine(log.ToString());
        }
    }
}
