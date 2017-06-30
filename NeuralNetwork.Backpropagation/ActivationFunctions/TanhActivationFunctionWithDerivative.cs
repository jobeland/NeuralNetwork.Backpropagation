using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class TanhActivationFunctionWithDerivative : TanhActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            //var function = new Derivative("tanh(x)");
            //return function.ComputeDerivative(signal);
            var val = CalculateActivation(signal);
            return 1 - val*val;
        }
    }
}
