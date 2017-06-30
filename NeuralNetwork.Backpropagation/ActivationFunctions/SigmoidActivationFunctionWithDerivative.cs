using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class SigmoidActivationFunctionWithDerivative : SigmoidActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            //var function = new Derivative("1/(1+exp(-x))");
            //return function.ComputeDerivative(signal);
            var val = CalculateActivation(signal);
            return val * (1 - val);
        }
    }
}
