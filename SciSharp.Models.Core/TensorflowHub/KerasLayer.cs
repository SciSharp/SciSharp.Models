using Tensorflow;
using Tensorflow.Keras.Engine;

namespace SciSharp.Models.TensorflowHub;

public class KerasLayer : Layer
{
    KerasLayerArgs _args;
    public KerasLayer(KerasLayerArgs args) : base(args)
    {
        _args = args;
        load_module(args.HandleName);
    }

    void load_module(string handle)
    {
        var module_path = @"C:\Users\haipi\AppData\Local\Temp\tfhub_modules\602d30248ff7929470db09f7385fc895e9ceb4c0";
        var model = Loader.load(module_path);
    }
}