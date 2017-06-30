using System;
using System.Diagnostics;
using System.Linq;
using ArtificialNeuralNetwork;
using ArtificialNeuralNetwork.Genes;
using NeuralNetwork.Backpropagation.ActivationFunctions;

namespace NeuralNetwork.Backpropagation.Models
{
    public class NeuronBackpropagationDecorator : IBackpropagationNeuron
    {
        public INeuron DecoratedNeuron { get; set; }
        public double Gradient { get; private set; }
        public bool IsDropped { get; set; }

        public void Process()
        {
            if (!IsDropped)
            {
                DecoratedNeuron.Process();
            }
        }

        public NeuronGene GetGenes()
        {
            return DecoratedNeuron.GetGenes();
        }

        public ISoma Soma
        {
            get { return DecoratedNeuron.Soma; }
            set { DecoratedNeuron.Soma = value; }
        }

        public IAxon Axon
        {
            get { return DecoratedNeuron.Axon; }
            set { DecoratedNeuron.Axon = value; }
        }

        public double CalculateGradient(double? target)
        {
            if (IsDropped)
            {
                Gradient = 0;
            }
            else
            {
                Gradient = DErrorTotalWithRespectToOutput(target) * DOutputWithRespectToNetInput();
            }
            return Gradient;
        }

        private double DErrorTotalWithRespectToOutput(double? target = null)
        {
            if (target == null)
            {
                return DecoratedNeuron.Axon.Terminals.Sum(s => s.Weight*((NeuronMappedSynapse) s).OutputNeuron.Gradient);
            }
            return -(target.Value - DecoratedNeuron.Axon.Value);
        }

        public void UpdateInputSynapseWeights(double learningRate, double momentum, int maxNormConstraint)
        {
            foreach (var synapse in Soma.Dendrites)
            {
                var decoratedSynapse = (NeuronMappedSynapse) synapse;
                if (decoratedSynapse.InputNeuron.IsDropped)
                {
                    //It's possible the output of the neuron feeding this synapse was dropped, so this synapse wouldn't affect error of this iteration
                    continue;
                }
                var dErrorWithRespectToWeight = Gradient*decoratedSynapse.InputNeuron.Axon.Value;
                var previousAdjustment = decoratedSynapse.WeightAdjustment;
                decoratedSynapse.WeightAdjustment = learningRate*dErrorWithRespectToWeight + momentum*previousAdjustment;
                decoratedSynapse.Weight = decoratedSynapse.Weight - decoratedSynapse.WeightAdjustment;
            }
            var weights = Soma.Dendrites.Select(s => s.Weight).ToList();
            var length = Math.Sqrt(weights.Sum(w => w*w));
            if (length > maxNormConstraint)
            {
                //constrain all weights to such that length will equal max-norm constraint
                var multipier = maxNormConstraint / length;
                foreach (var synapse in Soma.Dendrites)
                {
                    synapse.Weight = synapse.Weight*multipier;
                }
            }
        }

        private double DOutputWithRespectToNetInput()
        {
            var watch1 = Stopwatch.StartNew();
            var netInput = DecoratedNeuron.Soma.Value;
            var activationWithDerivative = (IActivationFunctionDerivative)DecoratedNeuron.Axon.ActivationFunction;
            watch1.Stop();
            var stamp1 = watch1.ElapsedTicks;
            var watch2 = Stopwatch.StartNew();
            var val1 = activationWithDerivative.CalculateDerivative(netInput);
            watch2.Stop();
            var stamp2 = watch2.ElapsedTicks;
            return val1;
        }

    }
}
