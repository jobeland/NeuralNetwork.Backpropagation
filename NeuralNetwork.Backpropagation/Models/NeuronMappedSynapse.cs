using ArtificialNeuralNetwork;

namespace NeuralNetwork.Backpropagation.Models
{
    public class NeuronMappedSynapse : Synapse
    {
        public IBackpropagationNeuron InputNeuron { get; set; }
        public IBackpropagationNeuron OutputNeuron { get; set; }
        public double WeightAdjustment { get; set; }
    }
}
