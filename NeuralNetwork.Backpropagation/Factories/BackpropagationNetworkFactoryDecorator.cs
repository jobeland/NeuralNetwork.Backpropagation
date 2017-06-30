using System.Collections.Generic;
using System.Linq;
using ArtificialNeuralNetwork;
using ArtificialNeuralNetwork.Factories;
using ArtificialNeuralNetwork.Genes;
using NeuralNetwork.Backpropagation.Models;

namespace NeuralNetwork.Backpropagation.Factories
{
    public class BackpropagationNetworkFactoryDecorator : INeuralNetworkFactory
    {
        private readonly INeuralNetworkFactory _decoratedFactory;

        public BackpropagationNetworkFactoryDecorator(INeuralNetworkFactory decoratedFactory)
        {
            _decoratedFactory = decoratedFactory;
        }

        public INeuralNetwork Create(int numInputs, int numOutputs, int numHiddenLayers, int numHiddenPerLayer)
        {
            var network = _decoratedFactory.Create(numInputs, numOutputs, numHiddenLayers, numHiddenPerLayer);
            SetUpSynapseMappingForPropogation(network);
            return network;
        }

        public INeuralNetwork Create(int numInputs, int numOutputs, IList<int> hiddenLayerSpecs)
        {
            var network = _decoratedFactory.Create(numInputs, numOutputs, hiddenLayerSpecs);
            SetUpSynapseMappingForPropogation(network);
            return network;

        }

        public INeuralNetwork Create(NeuralNetworkGene genes)
        {
            var network = _decoratedFactory.Create(genes);
            SetUpSynapseMappingForPropogation(network);
            return network;
        }


        internal void SetUpSynapseMappingForPropogation(INeuralNetwork network)
        {
            foreach (var neuron in network.InputLayer.NeuronsInLayer)
            {
                MapNeuronForSynapses((IBackpropagationNeuron)neuron);
            }
            foreach (var neuron in network.HiddenLayers.SelectMany(hl => hl.NeuronsInLayer))
            {
                MapNeuronForSynapses((IBackpropagationNeuron)neuron);
            }
            foreach (var neuron in network.OutputLayer.NeuronsInLayer)
            {
                MapNeuronForSynapses((IBackpropagationNeuron)neuron);
            }
        }

        internal void MapNeuronForSynapses(IBackpropagationNeuron neuron)
        {
            // loop through every layer, and for each neuron's axon's terminals, add that neuron as the input neuron
            // loop through every layer, and for each neuron's soma's dendrites, add that neuron as the output neuron
            foreach (var synapse in neuron.Axon.Terminals)
            {
                ((NeuronMappedSynapse)synapse).InputNeuron = neuron;
            }
            foreach (var synapse in neuron.Soma.Dendrites)
            {
                ((NeuronMappedSynapse)synapse).OutputNeuron = neuron;
            }
        }

    }
}
