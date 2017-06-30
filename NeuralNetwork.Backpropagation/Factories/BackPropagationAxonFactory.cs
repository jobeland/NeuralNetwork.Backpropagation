using System;
using System.Collections.Generic;
using ArtificialNeuralNetwork;
using ArtificialNeuralNetwork.ActivationFunctions;
using ArtificialNeuralNetwork.Factories;
using NeuralNetwork.Backpropagation.Models;

namespace NeuralNetwork.Backpropagation.Factories
{
    public class BackpropagationAxonFactory : IAxonFactory
    {
        private readonly IActivationFunction _activationFunction;

        private BackpropagationAxonFactory(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
        }

        public static IAxonFactory GetInstance(IActivationFunction activationFunction)
        {
            return new BackpropagationAxonFactory(activationFunction);
        }

        public IAxon Create(IList<Synapse> terminals)
        {
            return new AxonBackpropagation(terminals, _activationFunction);
        }

        public IAxon Create()
        {
            return new AxonBackpropagation(new List<Synapse>(), _activationFunction);
        }

        public IAxon Create(IList<Synapse> terminals, Type activationFunction)
        {
            var functionObj = Activator.CreateInstance(activationFunction);
            if (!(functionObj is IActivationFunction))
            {
                throw new NotSupportedException(
                    $"{activationFunction} is not a supported activation function type for Create() as it does not implement IActivationFunction");
            }
            var function = functionObj as IActivationFunction;
            return new AxonBackpropagation(terminals, function);
        }
    }
}
