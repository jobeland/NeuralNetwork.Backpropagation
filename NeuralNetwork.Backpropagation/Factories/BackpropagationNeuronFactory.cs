using ArtificialNeuralNetwork;
using ArtificialNeuralNetwork.Factories;
using NeuralNetwork.Backpropagation.Models;

namespace NeuralNetwork.Backpropagation.Factories
{
    public class BackpropagationNeuronFactory : INeuronFactory
    {
        private readonly INeuronFactory _neuronFactory;

        private BackpropagationNeuronFactory(INeuronFactory neuronFactory)
        {
            _neuronFactory = neuronFactory;
        }

        public static INeuronFactory GetInstance(INeuronFactory neuronFactory)
        {
            return new BackpropagationNeuronFactory(neuronFactory);
        }

        public INeuron Create(ISoma soma, IAxon axon)
        {
            return new NeuronBackpropagationDecorator
            {
                DecoratedNeuron = _neuronFactory.Create(soma, axon),
                IsDropped = false
            };
        }
    }
}
