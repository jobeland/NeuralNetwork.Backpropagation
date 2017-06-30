using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class LeakyRectifiedLinearActivationFunctionWithDerivative : LeakyRectifiedLinearActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            return signal > 0 ? 1 : 0.01;
        }
    }
}
