using ArtificialNeuralNetwork.ActivationFunctions;
using Rychusoft.NumericalLibraries.Derivative;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class SinhActivationFunctionWithDerivative : SinhActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            var function = new Derivative("sinh(x)");
            return function.ComputeDerivative(signal);
        }
    }
}
