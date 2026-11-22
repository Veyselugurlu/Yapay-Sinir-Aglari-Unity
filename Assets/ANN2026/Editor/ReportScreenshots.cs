using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditorInternal;
using UnityEngine;

/// <summary>
/// Takes the Unity Editor screenshots used in the report (menu: ANN / Take Report Screenshots, or create
/// Temp/report_capture.request while the editor is open). Walks through the steps below, enters Play mode
/// where needed and saves the main editor window as PNG (PrintWindow). Inspector changes made here are
/// temporary: the scene is re-opened without saving at the end.
/// State survives the domain reload of entering Play mode through SessionState.
/// </summary>
[InitializeOnLoad]
public static class ReportScreenshots
{
    private const string RequestFile = "Temp/report_capture.request";
    private const string DoneFile = "Temp/report_capture.done";
    private const string OutDir = "docs/week09/images/editor";
    private const string StepKey = "AnnShots.Step", TimeKey = "AnnShots.T0", EnteredKey = "AnnShots.Entered";

    // step: what to show                                          Play   wait (s)
    //  1 Brain scene, Brain settings in the Inspector             no     3
    //  2 Play: Console with SSE and the four XOR results          yes    2
    //  3 stop, set the activation to Step (not saved)             no     2
    //  4 Play: step function, SSE stays high                      yes    2
    //  5 stop, set alpha 0.1 (not saved)                          no     2
    //  6 Play: alpha 0.1 after 1000 epochs                        yes    2
    //  7 stop, open the Visualizer scene                          no     2
    //  8 Play: visualizer after 1000 epochs                       yes    see Tick
    //  9 stop, re-open the Brain scene without saving             no     2
    private static readonly float[] Wait = { 0f, 3f, 2f, 2f, 2f, 2f, 2f, 2f, 1f, 2f };

    static ReportScreenshots()
    {
        EditorApplication.update += Tick;
    }

    [MenuItem("ANN/Take Report Screenshots")]
    public static void Begin()
    {
        Directory.CreateDirectory(OutDir);
        if (File.Exists(DoneFile)) File.Delete(DoneFile);
        SessionState.SetInt(StepKey, 1);
        SessionState.SetBool(EnteredKey, false);
    }

    private static double Now => EditorApplication.timeSinceStartup;

    private static void Tick()
    {
        if (File.Exists(RequestFile))
        {
            File.Delete(RequestFile);
            Begin();
        }
        int step = SessionState.GetInt(StepKey, 0);
        if (step == 0) return;
        if (step == 1 && EditorApplication.isPlayingOrWillChangePlaymode)
        {
            EditorApplication.isPlaying = false;
            return;
        }
        if (!SessionState.GetBool(EnteredKey, false))
        {
            SessionState.SetBool(EnteredKey, true);
            SessionState.SetFloat(TimeKey, (float)Now);
            try { Enter(step); } catch (Exception e) { Fail(e); }
            return;
        }

        bool wantsPlay = step == 2 || step == 4 || step == 6 || step == 8;
        bool wantsStop = step == 3 || step == 5 || step == 7 || step == 9;
        if ((wantsPlay && !EditorApplication.isPlaying) || (wantsStop && EditorApplication.isPlayingOrWillChangePlaymode)
            || EditorApplication.isCompiling)
        {
            SessionState.SetFloat(TimeKey, (float)Now);
            return;
        }
        InternalEditorUtility.RepaintAllViews(); // an editor behind other windows does not repaint on its own
        if (EditorApplication.isPlaying && Time.timeSinceLevelLoad < 1f) { SessionState.SetFloat(TimeKey, (float)Now); return; }
        if (step == 8)
        {
            var v = UnityEngine.Object.FindAnyObjectByType<XorVisualizer>();
            if (v != null && Time.frameCount < 1000 / v.epochsPerFrame + 30) { SessionState.SetFloat(TimeKey, (float)Now); return; }
        }
        if (Now - SessionState.GetFloat(TimeKey, 0f) < Wait[step]) return;

        try { Leave(step); } catch (Exception e) { Fail(e); return; }
        int next = step + 1 < Wait.Length ? step + 1 : 0;
        SessionState.SetInt(StepKey, next);
        SessionState.SetBool(EnteredKey, false);
        if (next == 0)
        {
            File.WriteAllText(DoneFile, "ok");
            Debug.Log("ANN: report screenshots saved to " + OutDir);
        }
    }

    private static void Enter(int step)
    {
        switch (step)
        {
            case 1:
                EditorApplication.ExecuteMenuItem("Window/Layouts/Default");
                ClearConsole();
                EditorSceneManager.OpenScene(AnnSceneBuilder.BrainScenePath, OpenSceneMode.Single);
                Selection.activeGameObject = GameObject.Find("Brain");
                break;
            case 2:
            case 4:
            case 6:
                ClearConsole();
                EditorApplication.ExecuteMenuItem("Window/General/Console");
                Selection.activeGameObject = GameObject.Find("Brain");
                EditorApplication.isPlaying = true;
                break;
            case 3:
                EditorApplication.isPlaying = false;
                break;
            case 5:
                EditorApplication.isPlaying = false;
                break;
            case 7:
                EditorApplication.isPlaying = false;
                break;
            case 8:
                FocusGameView();
                EditorApplication.isPlaying = true;
                break;
            case 9:
                EditorApplication.isPlaying = false;
                break;
        }
    }

    private static void Leave(int step)
    {
        switch (step)
        {
            case 1: Capture("01_brain_inspector.png"); break;
            case 2: Capture("02_brain_console_xor.png"); break;
            case 3: SetBrain(b => { b.FindProperty("hiddenActivation").enumValueIndex = (int)Activation.Step;
                                    b.FindProperty("outputActivation").enumValueIndex = (int)Activation.Step; }); break;
            case 4: Capture("03_brain_console_step.png"); break;
            case 5: SetBrain(b => { b.FindProperty("hiddenActivation").enumValueIndex = (int)Activation.Sigmoid;
                                    b.FindProperty("outputActivation").enumValueIndex = (int)Activation.Sigmoid;
                                    b.FindProperty("alpha").doubleValue = 0.1; }); break;
            case 6: Capture("04_brain_console_alpha01.png"); break;
            case 7: EditorSceneManager.OpenScene(AnnSceneBuilder.VisualScenePath, OpenSceneMode.Single); break;
            case 8: Capture("05_visualizer_play.png"); break;
            case 9:
                EditorSceneManager.OpenScene(AnnSceneBuilder.BrainScenePath, OpenSceneMode.Single); // discards the temporary changes
                break;
        }
    }

    /// <summary>Changes the Brain component in the open scene; the scene is never saved.</summary>
    private static void SetBrain(Action<SerializedObject> change)
    {
        var brain = UnityEngine.Object.FindAnyObjectByType<Brain>();
        var so = new SerializedObject(brain);
        change(so);
        so.ApplyModifiedPropertiesWithoutUndo();
        Selection.activeGameObject = brain.gameObject;
    }

    private static void FocusGameView()
    {
        var gameViewType = Type.GetType("UnityEditor.GameView,UnityEditor");
        if (gameViewType != null) EditorWindow.GetWindow(gameViewType, false, null, true);
    }

    private static void ClearConsole()
    {
        Type.GetType("UnityEditor.LogEntries,UnityEditor")?.GetMethod("Clear")?.Invoke(null, null);
    }

    [StructLayout(LayoutKind.Sequential)] private struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)]
    private struct BITMAPINFOHEADER
    {
        public int biSize, biWidth, biHeight; public short biPlanes, biBitCount;
        public int biCompression, biSizeImage, biXPelsPerMeter, biYPelsPerMeter, biClrUsed, biClrImportant;
    }
    [DllImport("user32.dll")] private static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] private static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] private static extern IntPtr GetDC(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern int ReleaseDC(IntPtr hWnd, IntPtr hdc);
    [DllImport("gdi32.dll")] private static extern IntPtr CreateCompatibleDC(IntPtr hdc);
    [DllImport("gdi32.dll")] private static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int w, int h);
    [DllImport("gdi32.dll")] private static extern IntPtr SelectObject(IntPtr hdc, IntPtr obj);
    [DllImport("gdi32.dll")] private static extern bool DeleteObject(IntPtr obj);
    [DllImport("gdi32.dll")] private static extern bool DeleteDC(IntPtr hdc);
    [DllImport("gdi32.dll")] private static extern int GetDIBits(IntPtr hdc, IntPtr bmp, uint start, uint lines, byte[] bits, ref BITMAPINFOHEADER info, uint usage);

    /// <summary>
    /// Copies the main editor window with PrintWindow, which also works while the window is behind other
    /// windows, so the screenshots never take the screen away from whoever is using the computer.
    /// </summary>
    private static void Capture(string file)
    {
        IntPtr hwnd = System.Diagnostics.Process.GetCurrentProcess().MainWindowHandle;
        if (hwnd == IntPtr.Zero || !GetWindowRect(hwnd, out RECT r))
            throw new InvalidOperationException("Editor window not found; screenshot not taken: " + file);
        int w = r.Right - r.Left, h = r.Bottom - r.Top;
        IntPtr screen = GetDC(IntPtr.Zero), mem = CreateCompatibleDC(screen), bmp = CreateCompatibleBitmap(screen, w, h);
        IntPtr old = SelectObject(mem, bmp);
        try
        {
            if (!PrintWindow(hwnd, mem, 2)) // PW_RENDERFULLCONTENT
                throw new InvalidOperationException("PrintWindow failed: " + file);
            var info = new BITMAPINFOHEADER { biSize = Marshal.SizeOf<BITMAPINFOHEADER>(), biWidth = w, biHeight = h, biPlanes = 1, biBitCount = 32 };
            var bgra = new byte[w * h * 4];
            SelectObject(mem, old);
            GetDIBits(mem, bmp, 0, (uint)h, bgra, ref info, 0); // bottom-up rows, as Texture2D expects
            var rgba = new Color32[w * h];
            for (int i = 0; i < rgba.Length; i++)
                rgba[i] = new Color32(bgra[i * 4 + 2], bgra[i * 4 + 1], bgra[i * 4], 255);
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
            tex.SetPixels32(rgba);
            tex.Apply();
            File.WriteAllBytes(Path.Combine(OutDir, file), tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
        }
        finally
        {
            DeleteObject(bmp);
            DeleteDC(mem);
            ReleaseDC(IntPtr.Zero, screen);
        }
        Debug.Log("ANN: captured " + file);
    }

    private static void Fail(Exception e)
    {
        SessionState.SetInt(StepKey, 0);
        if (EditorApplication.isPlaying) EditorApplication.isPlaying = false;
        File.WriteAllText(DoneFile, "error: " + e);
        Debug.LogException(e);
    }
}
