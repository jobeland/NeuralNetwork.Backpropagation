using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ArtificialNeuralNetwork;
using Newtonsoft.Json;

namespace NeuralNetwork.Backpropagation
{
    public class BackpropagationTrainer : IBackpropagationTrainer
    {
        private readonly INeuralNetworkDropoutModifier _dropoutModifier;
        private readonly IBackpropagater _backpropagater;
        public INeuralNetwork Network { get; }

        public BackpropagationTrainer(INeuralNetworkDropoutModifier dropoutModifier, IBackpropagater backpropagater, INeuralNetwork network)
        {
            _dropoutModifier = dropoutModifier;
            _backpropagater = backpropagater;
            Network = network;
        }

        public void Train(ICollection<DataSet> dataSets, double minimumError, string storageDirectory)
        {
            var error = 1.0;
            var numEpochs = 0;
            if (!Directory.Exists(storageDirectory))
            {
                Directory.CreateDirectory(storageDirectory);
            }

            while (error > minimumError && numEpochs < int.MaxValue)
            {
                var errors = new List<double>();
                foreach (var dataSet in dataSets)
                {
                    _dropoutModifier.DropNeurons(Network);
                    Network.SetInputs(dataSet.Values);
                    Network.Process();
                    var totalError = _backpropagater.Backpropagate(dataSet.Targets);
                    errors.Add(totalError);
                }
                error = errors.Average();
                numEpochs++;
                Console.WriteLine($"Epoch: {numEpochs} \t Error: {error}");
                if (numEpochs % 10 == 0)
                {
                    var jsonNet = JsonConvert.SerializeObject(this, new JsonSerializerSettings { PreserveReferencesHandling = PreserveReferencesHandling.Objects });
                    File.WriteAllText($@"{storageDirectory}\backprop-{error}-{DateTime.Now.Ticks}.json", jsonNet);
                }
            }
        }
    }
}
