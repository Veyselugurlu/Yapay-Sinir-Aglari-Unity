using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Activation { Sigmoid, Step, Tanh, ReLU, LeakyReLU, Linear }

public class ANN{

    public int numberOfInputs;
    public int numberOfOutputs;
    public int numberOfHiddenLayers;
    public int numberOfNeuronsPerHiddenLayer;
    public double alpha;
    // Sigmoid everywhere reproduces the lecture network; the others are for the activation experiments.
    public Activation hiddenActivation = Activation.Sigmoid;
    public Activation outputActivation = Activation.Sigmoid;
    List<Layer> layers = new List<Layer>();

    public ANN(int numberOfInputs, int numberOfOutputs,
               int numberOfHiddenLayers,
               int numberOfNeuronsPerHiddenLayer,
               double alpha)
    {
        this.numberOfInputs = numberOfInputs;
        this.numberOfOutputs = numberOfOutputs;
        this.numberOfHiddenLayers = numberOfHiddenLayers;
        this.numberOfNeuronsPerHiddenLayer = numberOfNeuronsPerHiddenLayer;
        this.alpha = alpha;

        if(numberOfHiddenLayers > 0)
        {
            layers.Add(new Layer(numberOfNeuronsPerHiddenLayer, numberOfInputs));

            for(int i = 0; i < numberOfHiddenLayers-1; i++)
            {
                layers.Add(new Layer(numberOfNeuronsPerHiddenLayer, numberOfNeuronsPerHiddenLayer));
            }

            layers.Add(new Layer(numberOfOutputs, numberOfNeuronsPerHiddenLayer));
        }
        else
        {
            layers.Add(new Layer(numberOfOutputs, numberOfInputs));
        }
    }

    /// <summary>Forward pass; with updateWeights false the network only answers and does not learn.</summary>
    public List<double> Run(List<double> inputValues, List<double> desiredOutput, bool updateWeights = true)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if(inputValues.Count != numberOfInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numberOfInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);
        for(int i = 0; i < numberOfHiddenLayers + 1; i++)
        {
            if(i > 0)
            {
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            for(int j = 0; j < layers[i].numberOfNeurons; j++)
            {
                double N = 0;
                layers[i].neurons[j].inputs.Clear();

                for(int k = 0; k < layers[i].neurons[j].numberOfInputs; k++)
                {
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    N += layers[i].neurons[j].weights[k] * inputs[k];
                }

                N -= layers[i].neurons[j].bias;
                layers[i].neurons[j].output = ActivationFunction(N, i);
                outputs.Add(layers[i].neurons[j].output);
            }
        }

        if(updateWeights) UpdateWeights(outputs, desiredOutput);

        return outputs;
    }

    void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        for(int i = numberOfHiddenLayers; i >= 0; i--)
        {
            for(int j = 0; j < layers[i].numberOfNeurons; j++)
            {
                if(i == numberOfHiddenLayers)
                {
                    error = desiredOutput[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = Derivative(outputs[j], i) * error;
                    //errorGradient calculated with Delta Rule: en.wikipedia.org/wiki/Delta_rule
                }
                else
                {
                    layers[i].neurons[j].errorGradient = Derivative(layers[i].neurons[j].output, i);
                    double errorGradSum = 0;
                    for(int p = 0; p < layers[i+1].numberOfNeurons; p++)
                    {
                        errorGradSum += layers[i+1].neurons[p].errorGradient *
                                        layers[i+1].neurons[p].weights[j];
                    }
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }
                for(int k = 0; k < layers[i].neurons[j].numberOfInputs; k++)
                {
                    if(i == numberOfHiddenLayers)
                    {
                        error = desiredOutput[j] - outputs[j];
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] *
                                                           error;
                    }
                    else
                    {
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] *
                                                           layers[i].neurons[j].errorGradient;
                    }
                }
                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }

    //for full list of activation functions
    //see en.wikipedia.org/wiki/Activation_function
    double ActivationFunction(double value, int layer)
    {
        switch(layer == numberOfHiddenLayers ? outputActivation : hiddenActivation)
        {
            case Activation.Step: return Step(value);
            case Activation.Tanh: return System.Math.Tanh(value);
            case Activation.ReLU: return value > 0 ? value : 0;
            case Activation.LeakyReLU: return value > 0 ? value : 0.01 * value;
            case Activation.Linear: return value;
            default: return Sigmoid(value);
        }
    }

    /// <summary>
    /// Slope of the activation function, written in terms of its output as the lecture does for the
    /// sigmoid: output * (1 - output). The step function is flat everywhere, so its slope is 0.
    /// </summary>
    double Derivative(double output, int layer)
    {
        switch(layer == numberOfHiddenLayers ? outputActivation : hiddenActivation)
        {
            case Activation.Step: return 0;
            case Activation.Tanh: return 1 - output * output;
            case Activation.ReLU: return output > 0 ? 1 : 0;
            case Activation.LeakyReLU: return output > 0 ? 1 : 0.01;
            case Activation.Linear: return 1;
            default: return output * (1 - output);
        }
    }

    double Step(double value) //(aka binary step)
    {
        if(value < 0) return 0;
        else return 1;
    }

    double Sigmoid(double value) //(aka logistic softstep)
    {
        double k = (double) System.Math.Exp(value);
        return k / (1.0f + k);
    }
}
