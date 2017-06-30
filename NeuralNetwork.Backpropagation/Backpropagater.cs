using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using ArtificialNeuralNetwork;
using NeuralNetwork.Backpropagation.Models;

namespace NeuralNetwork.Backpropagation
{
    public class Backpropagater : IBackpropagater
    {
        private readonly INeuralNetwork _network;
        private readonly double _learningRate;
        private readonly double _momentum;
        private readonly int _maxNormConstraint;
        private readonly bool _useMultithreading;

        public Backpropagater(INeuralNetwork network, double learningRate, double momentum, int maxNormConstraint, bool useMultithreading)
        {
            _network = network;
            _learningRate = learningRate;
            _momentum = momentum;
            _maxNormConstraint = maxNormConstraint;
            _useMultithreading = useMultithreading;
        }

        public double Backpropagate(double[] targets)
        {
            var outputs = _network.GetOutputs();
            if (outputs.Length != targets.Length)
            {
                throw new Exception("output length does not match target length");
            }

            var totalError = outputs.Select((t, i) => CalculateError(t, targets[i])).Sum();

            //use https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ to help with equations and steps
            var watch1 = Stopwatch.StartNew();
            CalculateGradients(targets);
            watch1.Stop();
            var stamp1 = watch1.ElapsedTicks;
            var watch2 = Stopwatch.StartNew();
            UpdateWeights();
            watch2.Stop();
            var stamp2 = watch2.ElapsedTicks;
            return totalError;
        }

        internal void CalculateGradients(double[] targets)
        {
            for (var i = 0; i < targets.Length; i++)
            {
                var outputNeuron = _network.OutputLayer.NeuronsInLayer[i];
                ((IBackpropagationNeuron)outputNeuron).CalculateGradient(targets[i]);
            }
            for (var i = _network.HiddenLayers.Count - 1; i >= 0; i--)
            {
                var hiddenLayer = _network.HiddenLayers[i];
                if (_useMultithreading)
                {
                    Parallel.ForEach(hiddenLayer.NeuronsInLayer, (hiddenNeuron) =>
                    {
                        var backPropNeuron = (IBackpropagationNeuron)hiddenNeuron;
                        if (!backPropNeuron.IsDropped)
                        {
                            backPropNeuron.CalculateGradient();
                        }
                    });
                }
                else
                {
                    foreach (var hiddenNeuron in hiddenLayer.NeuronsInLayer)
                    {
                        var backPropNeuron = (IBackpropagationNeuron)hiddenNeuron;
                        if (!backPropNeuron.IsDropped)
                        {
                            backPropNeuron.CalculateGradient();
                        }
                    }
                }
            }
        }

        internal void UpdateWeights()
        {
            if (_useMultithreading)
            {
                var neurons = new List<INeuron>();
                neurons.AddRange(_network.OutputLayer.NeuronsInLayer);
                neurons.AddRange(_network.HiddenLayers.SelectMany(hl => hl.NeuronsInLayer));

                Parallel.ForEach(neurons, (neuron) =>
                {
                    var backPropNeuron = (IBackpropagationNeuron)neuron;
                    if (!backPropNeuron.IsDropped)
                    {
                        backPropNeuron.UpdateInputSynapseWeights(_learningRate, _momentum, _maxNormConstraint);
                    }
                });
            }
            else
            {
                foreach (var outputNeuron in _network.OutputLayer.NeuronsInLayer)
                {
                    ((IBackpropagationNeuron)outputNeuron).UpdateInputSynapseWeights(_learningRate, _momentum, _maxNormConstraint);
                }
                foreach (var hiddenLayer in _network.HiddenLayers)
                {
                    foreach (var hiddenNeuron in hiddenLayer.NeuronsInLayer)
                    {
                        var backPropNeuron = (IBackpropagationNeuron)hiddenNeuron;
                        if (!backPropNeuron.IsDropped)
                        {
                            backPropNeuron.UpdateInputSynapseWeights(_learningRate, _momentum, _maxNormConstraint);
                        }
                    }
                }
            }
        }

        internal double CalculateError(double output, double target)
        {
            return (target - output) * (target - output) / 2;
        }
    }
}
