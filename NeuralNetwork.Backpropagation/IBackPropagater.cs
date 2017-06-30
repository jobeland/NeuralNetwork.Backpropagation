namespace NeuralNetwork.Backpropagation
{
    public interface IBackpropagater
    {
        double Backpropagate(double[] targets);
    }
}