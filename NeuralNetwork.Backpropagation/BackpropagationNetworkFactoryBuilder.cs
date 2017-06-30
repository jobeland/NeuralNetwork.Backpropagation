using ArtificialNeuralNetwork.ActivationFunctions;
using ArtificialNeuralNetwork.Factories;
using ArtificialNeuralNetwork.WeightInitializer;
using NeuralNetwork.Backpropagation.ActivationFunctions;
using NeuralNetwork.Backpropagation.Factories;

namespace NeuralNetwork.Backpropagation
{
    public class BackpropagationNetworkFactoryBuilder : IBackpropagationNetworkFactoryBuilder
    {
        public INeuralNetworkFactory BuildBackpropagationNetworkFactory(IWeightInitializer weightInitializer,
            ISomaFactory somaFactory,
            IActivationFunctionDerivative activationFunctionDerivative,
            IActivationFunction inputActivationFunction,
            INeuronFactory neuronFactory)
        {
            var axonFactory = BackpropagationAxonFactory.GetInstance(activationFunctionDerivative);
            var hiddenSynapseFactory = DecoratedSynapseFactory.GetInstance(weightInitializer,
                AxonFactory.GetInstance(activationFunctionDerivative));
            var inputSynapseFactory = DecoratedSynapseFactory.GetInstance(new ConstantWeightInitializer(1.0),
                AxonFactory.GetInstance(inputActivationFunction));
            var decoratedNeuronFactory = BackpropagationNeuronFactory.GetInstance(neuronFactory);
            INeuralNetworkFactory factory = NeuralNetworkFactory.GetInstance(somaFactory, axonFactory,
                hiddenSynapseFactory, inputSynapseFactory, weightInitializer, decoratedNeuronFactory);

            var backPropNetworkFactory = new BackpropagationNetworkFactoryDecorator(factory);
            return backPropNetworkFactory;
        }
    }
}
