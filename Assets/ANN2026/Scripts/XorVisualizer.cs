using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

/// <summary>
/// Trains its own network a few epochs per frame and draws what it answers for every point of the
/// input square [0,1] x [0,1]: blue = 1, orange = 0, grey = undecided. The four training points are
/// drawn on top, so the viewer can watch the network carve out the XOR regions.
/// Command line (build): -seed 10  -recordFrames &lt;dir&gt;  -recordCount 300
/// </summary>
public class XorVisualizer : MonoBehaviour
{
    public Gate gate = Gate.XOR;
    public int seed = 10;
    public double alpha = 0.8;
    [Min(1)] public int epochsPerFrame = 4;
    [Min(1)] public int maxEpochs = 1000;

    private const int Size = 64;
    private static readonly Color One = new Color32(0x2a, 0x78, 0xd6, 0xff);
    private static readonly Color Zero = new Color32(0xeb, 0x68, 0x34, 0xff);
    private static readonly Color Mid = new Color32(0xe4, 0xe3, 0xde, 0xff);

    private ANN ann;
    private double[][] set;
    private Texture2D surface;
    private int epoch;
    private double sse;
    private readonly double[] outs = new double[4];
    private GUIStyle style, small;

    private void Start()
    {
        string[] args = Environment.GetCommandLineArgs();
        if (int.TryParse(Value(args, "-seed"), out int s)) seed = s;
        UnityEngine.Random.InitState(seed);
        ann = new ANN(2, 1, 1, 2, alpha);
        set = Brain.TrainingSet(gate);
        surface = new Texture2D(Size, Size, TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
        Redraw();

        string dir = Value(args, "-recordFrames");
        if (!string.IsNullOrEmpty(dir))
        {
            Application.runInBackground = true;
            Time.captureDeltaTime = 1f / 20f;
            int count = int.TryParse(Value(args, "-recordCount"), out int c) ? c : 300;
            StartCoroutine(Record(dir, count));
        }
    }

    private void Update()
    {
        if (epoch >= maxEpochs) return;
        for (int i = 0; i < epochsPerFrame && epoch < maxEpochs; i++, epoch++)
            sse = Brain.TrainEpoch(ann, set);
        Redraw();
    }

    private double Predict(double x, double y) =>
        ann.Run(new List<double> { x, y }, new List<double> { 0 }, false)[0];

    private void Redraw()
    {
        var pixels = new Color[Size * Size];
        for (int py = 0; py < Size; py++)
            for (int px = 0; px < Size; px++)
            {
                float v = (float)Predict(px / (Size - 1.0), py / (Size - 1.0));
                if (float.IsNaN(v)) v = 0.5f;
                pixels[py * Size + px] = v >= 0.5f ? Color.Lerp(Mid, One, (v - 0.5f) * 2f) : Color.Lerp(Zero, Mid, v * 2f);
            }
        surface.SetPixels(pixels);
        surface.Apply();
        for (int k = 0; k < 4; k++) outs[k] = Predict(set[k][0], set[k][1]);
    }

    private void OnGUI()
    {
        if (style == null)
        {
            style = new GUIStyle(GUI.skin.label) { fontSize = 18 };
            style.normal.textColor = Color.white;
            small = new GUIStyle(style) { fontSize = 15, alignment = TextAnchor.MiddleCenter };
        }
        float side = Mathf.Min(Screen.height - 80, Screen.width * 0.55f);
        var r = new Rect(30, 40, side, side);
        GUI.DrawTexture(r, surface, ScaleMode.StretchToFill);
        // texture row 0 is input2 = 0, drawn at the bottom
        for (int k = 0; k < 4; k++)
        {
            float cx = r.x + (float)set[k][0] * r.width, cy = r.yMax - (float)set[k][1] * r.height;
            Color previous = GUI.color;
            GUI.color = set[k][2] > 0.5 ? One : Zero;
            GUI.DrawTexture(new Rect(cx - 11, cy - 11, 22, 22), Texture2D.whiteTexture);
            GUI.color = previous;
            GUI.Label(new Rect(cx - 40, cy + (set[k][1] > 0.5 ? 14 : -40), 80, 24), $"({set[k][0]},{set[k][1]})", small);
        }
        GUI.Label(new Rect(30, 10, side, 26), "input 1 →  ·  input 2 ↑", style);

        float x = r.xMax + 30;
        GUI.Label(new Rect(x, 40, 420, 28), $"{gate}  ·  2-2-1 network  ·  alpha {alpha}", style);
        GUI.Label(new Rect(x, 72, 420, 28), $"Epoch {epoch} / {maxEpochs}", style);
        GUI.Label(new Rect(x, 104, 420, 28), $"Sum of squared error: {sse:0.0000}", style);
        for (int k = 0; k < 4; k++)
            GUI.Label(new Rect(x, 150 + 30 * k, 420, 28),
                $"{set[k][0]} {set[k][1]}  ->  {outs[k]:0.000}  ({Math.Round(outs[k])}, expected {set[k][2]})", style);
        GUI.Label(new Rect(x, 290, 420, 28), "blue = 1, orange = 0", style);
    }

    private static IEnumerator Record(string dir, int count)
    {
        Directory.CreateDirectory(dir);
        for (int saved = 0; saved < count; saved++)
        {
            yield return new WaitForEndOfFrame();
            Texture2D shot = ScreenCapture.CaptureScreenshotAsTexture();
            File.WriteAllBytes(Path.Combine(dir, $"frame_{saved:0000}.png"), shot.EncodeToPNG());
            Destroy(shot);
        }
        Application.Quit();
    }

    private static string Value(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }
}
