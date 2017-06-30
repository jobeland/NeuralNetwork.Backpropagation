using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class StepActivationFunctionWithDerivative : StepActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            return signal == 0.0 ? double.PositiveInfinity : 0.0;
        }
    }
}
