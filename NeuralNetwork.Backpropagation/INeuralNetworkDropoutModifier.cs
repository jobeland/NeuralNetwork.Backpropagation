using ArtificialNeuralNetwork;

namespace NeuralNetwork.Backpropagation
{
    public interface INeuralNetworkDropoutModifier
    {
        void DropNeurons(INeuralNetwork network);
    }
}