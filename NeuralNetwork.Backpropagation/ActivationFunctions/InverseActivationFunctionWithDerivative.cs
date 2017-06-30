using ArtificialNeuralNetwork.ActivationFunctions;
using Rychusoft.NumericalLibraries.Derivative;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class InverseActivationFunctionWithDerivative : InverseActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            var function = new Derivative("1/x");
            return function.ComputeDerivative(signal);
        }
    }
}
