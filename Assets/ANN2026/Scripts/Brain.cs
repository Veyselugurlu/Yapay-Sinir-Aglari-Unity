using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum Gate { XOR, XNOR }

public class Brain : MonoBehaviour {

    [Header("Training set")]
    public Gate gate = Gate.XOR;

    [Header("Network (2 inputs, 1 output)")]
    [Min(0)] public int hiddenLayers = 1;
    [Min(1)] public int neuronsPerHiddenLayer = 2;
    public Activation hiddenActivation = Activation.Sigmoid;
    public Activation outputActivation = Activation.Sigmoid;

    [Header("Training")]
    [Tooltip("Training rate. The lecture uses 0.8; with 0.1 training is not finished after 1000 epochs.")]
    public double alpha = 0.8;
    [Min(1)] public int epochs = 1000;
    [Tooltip("Seeds the random initial weights so a run can be repeated; 0 = different every time.")]
    public int seed = 10;

    ANN ann;
    double sumSquareError = 0;

    /// <summary>The four rows {input1, input2, output} in the order the lecture trains them.</summary>
    public static double[][] TrainingSet(Gate gate)
    {
        double same = gate == Gate.XOR ? 0 : 1;   // XOR: 1 when inputs differ; XNOR: 1 when they are equal
        return new[]
        {
            new double[] { 1, 1, same }, new double[] { 1, 0, 1 - same },
            new double[] { 0, 1, 1 - same }, new double[] { 0, 0, same },
        };
    }

    /// <summary>One epoch: every row once through ANN.Run (which also updates the weights). Returns the sum of squared errors.</summary>
    public static double TrainEpoch(ANN ann, double[][] set)
    {
        double sse = 0;
        foreach (double[] row in set)
        {
            List<double> result = Train(ann, row[0], row[1], row[2]);
            sse += Mathf.Pow((float)result[0] - (float)row[2], 2);
        }
        return sse;
    }

    void Start () {
        if (seed != 0) Random.InitState(seed);
        ann = new ANN(2, 1, hiddenLayers, neuronsPerHiddenLayer, alpha)
        {
            hiddenActivation = hiddenActivation,
            outputActivation = outputActivation,
        };
        double[][] set = TrainingSet(gate);

        for(int i = 0; i < epochs; i++)
        {
            sumSquareError = TrainEpoch(ann, set);
        }
        Debug.Log(gate + " SSE: " + sumSquareError + (seed != 0 ? "  (seed " + seed + ")" : ""));

        foreach (double[] row in set)
        {
            List<double> result = ann.Run(new List<double> { row[0], row[1] }, new List<double> { row[2] }, false);
            Debug.Log(" " + row[0] + " " + row[1] + " " + result[0].ToString("0.0000") +
                      "  -> " + System.Math.Round(result[0]) + " (expected " + row[2] + ")");
        }
    }

    static List<double> Train(ANN ann, double i1, double i2, double o)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return (ann.Run(inputs, outputs));
    }
}
