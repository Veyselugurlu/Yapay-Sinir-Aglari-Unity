using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Trains the lecture network (ANN + Brain.TrainEpoch, the code the scene runs) under every condition of
/// the homework with the same 50 seeds and writes
///   docs/week09/data/runs.csv     one row per condition x seed: final SSE, outputs, learned or not
///   docs/week09/data/curves.csv   SSE every 20 epochs (every 200 for the long runs)
/// A run counts as learned when all four rounded outputs match the training set.
/// Menu: ANN / Run Experiments, batch: -executeMethod AnnExperiments.RunBatch
/// </summary>
public static class AnnExperiments
{
    private const string OutDir = "docs/week09/data";
    public const int Seeds = 50;

    private class Condition
    {
        public string id, group;
        public Gate gate = Gate.XOR;
        public double alpha = 0.8;
        public int epochs = 1000, layers = 1, neurons = 2;
        public Activation hidden = Activation.Sigmoid, output = Activation.Sigmoid;
    }

    private static List<Condition> Conditions()
    {
        var list = new List<Condition>
        {
            new Condition { id = "default", group = "default" },
            new Condition { id = "xnor", group = "gate", gate = Gate.XNOR },
            new Condition { id = "alpha0.1_10000", group = "long", alpha = 0.1, epochs = 10000 },
            new Condition { id = "default_10000", group = "long", epochs = 10000 },
            new Condition { id = "step_all", group = "activation", hidden = Activation.Step, output = Activation.Step },
        };
        foreach (double a in new[] { 0.1, 0.4, 0.8, 1.5, 3, 10, 30, 100, 1000 })
            list.Add(new Condition { id = "alpha" + a.ToString(CultureInfo.InvariantCulture), group = "alpha", alpha = a });
        foreach (Activation act in new[] { Activation.Sigmoid, Activation.Tanh, Activation.ReLU, Activation.LeakyReLU, Activation.Linear, Activation.Step })
            foreach (double a in new[] { 0.1, 0.8 })
                list.Add(new Condition { id = $"hidden_{act}_a{a.ToString(CultureInfo.InvariantCulture)}", group = "activation", hidden = act, alpha = a });
        foreach (int n in new[] { 1, 2, 3, 4, 8 })
            list.Add(new Condition { id = $"neurons{n}", group = "neurons", neurons = n });
        foreach (int l in new[] { 0, 1, 2, 3 })
            list.Add(new Condition { id = $"layers{l}", group = "layers", layers = l });
        return list;
    }

    [MenuItem("ANN/Run Experiments")]
    public static void Run()
    {
        var runs = new StringBuilder("condition,group,gate,alpha,epochs,hidden_layers,neurons,hidden_act,output_act,seed," +
                                     "final_sse,nan,out0,out1,out2,out3,correct,learned,first_epoch_sse_below_0.01\n");
        var curves = new StringBuilder("condition,seed,epoch,sse\n");
        bool logWasEnabled = Debug.unityLogger.logEnabled;
        try
        {
            foreach (Condition c in Conditions())
            {
                double[][] set = Brain.TrainingSet(c.gate);
                int every = c.epochs > 1000 ? 200 : 20;
                for (int seed = 1; seed <= Seeds; seed++)
                {
                    Random.InitState(seed); // same seed -> same initial weights in every condition with the same shape
                    var ann = new ANN(2, 1, c.layers, c.neurons, c.alpha) { hiddenActivation = c.hidden, outputActivation = c.output };
                    double sse = 0;
                    int firstLow = -1;
                    for (int e = 1; e <= c.epochs; e++)
                    {
                        sse = Brain.TrainEpoch(ann, set);
                        if (firstLow < 0 && sse < 0.01) firstLow = e;
                        if (e % every == 0 || e == 1) curves.Append($"{c.id},{seed},{e},{F(sse)}\n");
                    }
                    var outs = new double[4];
                    int correct = 0;
                    Debug.unityLogger.logEnabled = false;
                    for (int k = 0; k < 4; k++)
                    {
                        outs[k] = ann.Run(new List<double> { set[k][0], set[k][1] }, new List<double> { set[k][2] }, false)[0];
                        if (!double.IsNaN(outs[k]) && System.Math.Round(outs[k]) == set[k][2]) correct++;
                    }
                    Debug.unityLogger.logEnabled = logWasEnabled;
                    bool nan = double.IsNaN(sse) || double.IsNaN(outs[0]);
                    runs.Append(string.Join(",", c.id, c.group, c.gate, F(c.alpha), c.epochs, c.layers, c.neurons, c.hidden, c.output, seed,
                        F(sse), nan ? 1 : 0, F(outs[0]), F(outs[1]), F(outs[2]), F(outs[3]), correct, correct == 4 ? 1 : 0,
                        firstLow < 0 ? "" : firstLow.ToString())).Append('\n');
                }
            }
        }
        finally
        {
            Debug.unityLogger.logEnabled = logWasEnabled;
        }
        Directory.CreateDirectory(OutDir);
        File.WriteAllText(Path.Combine(OutDir, "runs.csv"), runs.ToString(), new UTF8Encoding(false));
        File.WriteAllText(Path.Combine(OutDir, "curves.csv"), curves.ToString(), new UTF8Encoding(false));
        Debug.Log("ANN: experiments written to " + OutDir);
    }

    public static void RunBatch()
    {
        int code = 0;
        try { Run(); }
        catch (System.Exception e) { Debug.LogException(e); code = 1; }
        EditorApplication.Exit(code);
    }

    private static string F(double v) => double.IsNaN(v) ? "NaN" : v.ToString("0.######", CultureInfo.InvariantCulture);
}
