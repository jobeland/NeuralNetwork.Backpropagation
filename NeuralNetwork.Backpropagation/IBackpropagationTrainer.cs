using System.Collections.Generic;
using ArtificialNeuralNetwork;

namespace NeuralNetwork.Backpropagation
{
    public interface IBackpropagationTrainer
    {
        INeuralNetwork Network { get; }
        void Train(ICollection<DataSet> dataSets, double minimumError, string storageDirectory);
    }
}