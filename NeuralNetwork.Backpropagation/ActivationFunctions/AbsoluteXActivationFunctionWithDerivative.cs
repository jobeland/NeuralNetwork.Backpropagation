using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class AbsoluteXActivationFunctionWithDerivative : AbsoluteXActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            var derivative = new StepActivationFunction();
            return derivative.CalculateActivation(signal);
        }
    }
}
