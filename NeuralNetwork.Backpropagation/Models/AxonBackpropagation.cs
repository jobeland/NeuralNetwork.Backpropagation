using System.Collections.Generic;
using ArtificialNeuralNetwork;
using ArtificialNeuralNetwork.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.Models
{
    public class AxonBackpropagation : Axon
    {
        public double DropoutMultiplier { get; set; }
        public AxonBackpropagation(IList<Synapse> terminals, IActivationFunction activationFunction) :base(terminals,activationFunction)
        {
            //default no dropout
            DropoutMultiplier = 1;
        }

        public override void ProcessSignal(double signal)
        {
            base.ProcessSignal(signal);
            Value = Value*DropoutMultiplier;
        }
    }
}
