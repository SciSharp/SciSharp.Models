using Tensorflow.Keras;

namespace SciSharp.Models.TensorflowHub;

public class HubApis
{
    /// <summary>
    /// Wraps a SavedModel (or a legacy TF1 Hub format) as a Keras Layer.
    /// </summary>
    /// <param name="handle"></param>
    /// <returns></returns>
    public ILayer KerasLayer(string handle)
        => new KerasLayer(new KerasLayerArgs
        {
            HandleName = handle
        });
}
