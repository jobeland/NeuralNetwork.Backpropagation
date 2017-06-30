using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public interface IActivationFunctionDerivative : IActivationFunction
    {
        double CalculateDerivative(double signal);
    }
}
