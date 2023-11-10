using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace SciSharp.Models.ImageClassification
{
    /// <summary>
    /// a layer blocks 
    /// </summary>
    public class BlocksLayer : Layer
    {
        static int layerId;

        public BlocksLayer(LayerArgs args) : base(args)
        {

        }

        public BlocksLayer(IEnumerable<ILayer> layers) : base(new LayerArgs { Name = "BlocksLayer_" + ++layerId })
        {
            Layers.AddRange(layers);
        }

        public BlocksLayer(IEnumerable<ILayer> layers, string name) : base(new LayerArgs { Name = name })
        {
            Layers.AddRange(layers);
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // print($"layer.name {this.Name} x.shape:{inputs.shape}");
            var y = inputs;

            foreach (var lay in Layers)
            {
                y = lay.Apply(y, state, training, optional_args);
            }

            return y;
        }


        public override List<IVariableV1> TrainableVariables
        {
            get
            {
                var ret = new List<IVariableV1>();

                foreach (var lay in Layers)
                {
                    ret.AddRange(lay.TrainableVariables);
                }
                return ret;
            }
        }

        public override List<IVariableV1> TrainableWeights
        {
            get
            {
                var ret = new List<IVariableV1>();

                foreach (var lay in Layers)
                {
                    ret.AddRange(lay.TrainableWeights);
                }
                return ret;
            }
        }
    }
}
