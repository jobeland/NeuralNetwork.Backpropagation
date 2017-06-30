using System;
using System.Linq;
using ArtificialNeuralNetwork;
using NeuralNetwork.Backpropagation.Models;

namespace NeuralNetwork.Backpropagation
{
    public class NeuralNetworkDropoutModifier : INeuralNetworkDropoutModifier
    {
        private readonly double _inputRetainProbability;
        private readonly double _hiddenRetainProbability;
        private readonly Random _random;

        public NeuralNetworkDropoutModifier(double inputRetainProbability, double hiddenRetainProbability, int randomSeed)
        {
            _inputRetainProbability = inputRetainProbability;
            _hiddenRetainProbability = hiddenRetainProbability;
            _random = new Random(randomSeed);
        }

        public void DropNeurons(INeuralNetwork network)
        {
            foreach (var neuron in network.InputLayer.NeuronsInLayer)
            {
                if (_random.NextDouble() > _inputRetainProbability)
                {
                    var backPropNeuron = (IBackpropagationNeuron)neuron;
                    backPropNeuron.IsDropped = true;
                    var backPropAxon = (AxonBackpropagation)backPropNeuron.Axon;
                    backPropAxon.DropoutMultiplier = 0;
                }
                else
                {
                    var backPropNeuron = (IBackpropagationNeuron)neuron;
                    backPropNeuron.IsDropped = false;
                    var backPropAxon = (AxonBackpropagation)backPropNeuron.Axon;
                    backPropAxon.DropoutMultiplier = 1 / _inputRetainProbability;
                }
            }
            foreach (var neuron in network.HiddenLayers.SelectMany(l => l.NeuronsInLayer))
            {
                if (_random.NextDouble() > _hiddenRetainProbability)
                {
                    var backPropNeuron = (IBackpropagationNeuron)neuron;
                    backPropNeuron.IsDropped = true;
                    var backPropAxon = (AxonBackpropagation)backPropNeuron.Axon;
                    backPropAxon.DropoutMultiplier = 0;
                }
                else
                {
                    var backPropNeuron = (IBackpropagationNeuron)neuron;
                    backPropNeuron.IsDropped = false;
                    var backPropAxon = (AxonBackpropagation)backPropNeuron.Axon;
                    backPropAxon.DropoutMultiplier = 1 / _hiddenRetainProbability;
                }
            }
        }
    }
}
