using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.ActivationFunctions
{
    public class RectifiedLinearActivationFunctionWithDerivative : RectifiedLinearActivationFunction, IActivationFunctionDerivative
    {
        public double CalculateDerivative(double signal)
        {
            return signal > 0 ? 1 : 0.0;
        }
    }
}
