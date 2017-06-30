using ArtificialNeuralNetwork;

namespace NeuralNetwork.Backpropagation.Models
{
    public interface IBackpropagationNeuron : INeuron
    {
        double CalculateGradient(double? target = null);
        void UpdateInputSynapseWeights(double learningRate, double momentum, int maxNormConstraint);
        double Gradient { get; }
        bool IsDropped { get; set; }
    }
}
