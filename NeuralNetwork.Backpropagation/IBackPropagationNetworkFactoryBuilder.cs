using ArtificialNeuralNetwork.ActivationFunctions;
using ArtificialNeuralNetwork.Factories;
using ArtificialNeuralNetwork.WeightInitializer;
using NeuralNetwork.Backpropagation.ActivationFunctions;

namespace NeuralNetwork.Backpropagation
{
    public interface IBackpropagationNetworkFactoryBuilder
    {
        INeuralNetworkFactory BuildBackpropagationNetworkFactory(IWeightInitializer weightInitializer,
            ISomaFactory somaFactory,
            IActivationFunctionDerivative activationFunctionDerivative,
            IActivationFunction inputActivationFunction,
            INeuronFactory neuronFactory);
    }
}