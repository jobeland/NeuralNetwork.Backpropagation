using ArtificialNeuralNetwork.ActivationFunctions;
using Rychusoft.NumericalLibraries.Derivative;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class IdentityActivationFunctionWithDerivative : IdentityActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            var function = new Derivative("x");
            return function.ComputeDerivative(signal);
        }
    }
}
