using ArtificialNeuralNetwork;
using ArtificialNeuralNetwork.Factories;
using ArtificialNeuralNetwork.WeightInitializer;
using NeuralNetwork.Backpropagation.Models;

namespace NeuralNetwork.Backpropagation.Factories
{
    public class DecoratedSynapseFactory : ISynapseFactory
    {
        private IWeightInitializer _weightInitializer;
        private IAxonFactory _axonFactory;

        private DecoratedSynapseFactory(IWeightInitializer weightInitializer, IAxonFactory axonFactory)
        {
            _weightInitializer = weightInitializer;
            _axonFactory = axonFactory;
        }

        public static ISynapseFactory GetInstance(IWeightInitializer weightInitializer, IAxonFactory axonFactory)
        {
            return new DecoratedSynapseFactory(weightInitializer, axonFactory);
        }

        public Synapse Create()
        {
            return new NeuronMappedSynapse
            {
                Axon = _axonFactory.Create(),
                Weight = _weightInitializer.InitializeWeight()
            };
        }

        public Synapse Create(double weight)
        {
            return new NeuronMappedSynapse
            {
                Axon = _axonFactory.Create(),
                Weight = weight
            };
        }
    }
}
