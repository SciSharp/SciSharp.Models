using System;
using System.Collections.Generic;
using System.Text;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using Tensorflow.Keras.Layers;

namespace SciSharp.Models.Transformer
{
    public class TransformerModel
    {
        TransformerConfig cfg;
        TransformerDataset dataloader;

        ILayer embedding_layer;
        ILayer transformer_block;
        ILayer pooling;
        ILayer dropout1;
        ILayer dense;
        ILayer dropout2;
        ILayer reshape;
        ILayer output;

        public TransformerModel()
        {
            cfg = new TransformerConfig();
            dataloader = new TransformerDataset();
            Init();
        }

        public TransformerModel(TransformerConfig config)
        {
            cfg = config;
            dataloader = new TransformerDataset(config);
            Init();
        }
        void Init()
        {
            embedding_layer = new TokenAndPositionEmbedding(new TokenAndPositionEmbeddingArgs { Maxlen = cfg.DatasetConfig.maxlen, VocabSize = cfg.DatasetConfig.vocab_size, EmbedDim = cfg.Transformer.embed_dim });
            transformer_block = new TransformerBlock(new TransformerBlockArgs { EmbedDim = cfg.Transformer.embed_dim, NumHeads = cfg.Transformer.num_heads, FfDim = cfg.Transformer.ff_dim });
            //pooling = keras.layers.GlobalAveragePooling1D();
            //dropout1 = keras.layers.Dropout(cfg.Transformer.dropout_rate);
            //dense = keras.layers.Dense(cfg.Transformer.dense_dim, activation: "relu");
            //dropout2 = keras.layers.Dropout(cfg.Transformer.dropout_rate);
            reshape = keras.layers.Reshape((cfg.DatasetConfig.maxlen * cfg.Transformer.embed_dim));
            output = keras.layers.Dense(2, activation: "softmax");
        }
        public Tensor Apply(Tensor inputs)
        {
            var embeddings = embedding_layer.Apply(inputs);
            var outputs = transformer_block.Apply(embeddings);
            //outputs = pooling.Apply(outputs);
            //outputs = dropout1.Apply(outputs);
            //outputs = dense.Apply(outputs);
            //outputs = dropout2.Apply(outputs);
            outputs = reshape.Apply(outputs);
            outputs = output.Apply(outputs);
            return outputs;
        }
        public IModel Build()
        {
            var inputs = keras.layers.Input(shape: new[] { cfg.DatasetConfig.maxlen });
            var outputs = Apply(inputs);
            return keras.Model(inputs: inputs, outputs: outputs);
        }

        public ICallback Train()
        {
            var dataset = dataloader.GetData();
            var x_train = dataset[0];
            var y_train = dataset[1];
            var x_val = dataset[2];
            var y_val = dataset[3];
            var model = Build();
            model.summary();
            model.compile(optimizer: "adam", loss: "sparse_categorical_crossentropy", metrics: new string[] { "accuracy" });
            var history = model.fit((NDArray)x_train, (NDArray)y_train, batch_size: cfg.TRAIN.batch_size, epochs: cfg.TRAIN.epochs, validation_data: ((Tensorflow.NumPy.NDArray val_x, Tensorflow.NumPy.NDArray val_y))(x_val, y_val));
            return history;
        }
    }
}
