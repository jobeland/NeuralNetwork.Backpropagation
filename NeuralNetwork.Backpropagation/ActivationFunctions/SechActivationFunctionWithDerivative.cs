using ArtificialNeuralNetwork.ActivationFunctions;
using Rychusoft.NumericalLibraries.Derivative;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    class SechActivationFunctionWithDerivative : SechActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            var function = new Derivative("cosh(x)");
            return function.ComputeDerivative(signal);
        }
    }
}
