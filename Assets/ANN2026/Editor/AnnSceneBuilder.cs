using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Builds the scenes from code (menu: ANN).
///   Brain.unity        empty "Brain" object with the Brain script: trains XOR in Start, prints to the Console
///   Visualizer.unity   the same network trained a few epochs per frame and drawn as a coloured map
/// </summary>
public static class AnnSceneBuilder
{
    public const string BrainScenePath = "Assets/ANN2026/Scenes/Brain.unity";
    public const string VisualScenePath = "Assets/ANN2026/Scenes/Visualizer.unity";

    [MenuItem("ANN/Build Scenes")]
    public static void BuildAll()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        new GameObject("Brain").AddComponent<Brain>();
        Save(scene, BrainScenePath);

        scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        Camera.main.clearFlags = CameraClearFlags.SolidColor;
        Camera.main.backgroundColor = new Color(0.12f, 0.13f, 0.16f);
        new GameObject("Visualizer").AddComponent<XorVisualizer>();
        Save(scene, VisualScenePath);

        EditorBuildSettings.scenes = new[]
        {
            new EditorBuildSettingsScene(BrainScenePath, true),
            new EditorBuildSettingsScene(VisualScenePath, true),
        };
        if (File.Exists("Assets/Scenes/SampleScene.unity"))
        {
            AssetDatabase.DeleteAsset("Assets/Scenes/SampleScene.unity");
            if (Directory.GetFiles("Assets/Scenes").Length == 0) AssetDatabase.DeleteAsset("Assets/Scenes");
        }
    }

    private static void Save(UnityEngine.SceneManagement.Scene scene, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        EditorSceneManager.SaveScene(scene, path);
        Debug.Log("ANN: scene saved -> " + path);
    }

    [MenuItem("ANN/Build Visualizer Player")]
    public static bool BuildPlayer()
    {
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { VisualScenePath },
            locationPathName = "Builds/Visualizer/Visualizer.exe",
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None,
        });
        Debug.Log("ANN: build " + report.summary.result);
        return report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded;
    }

    /// <summary>-executeMethod AnnSceneBuilder.BuildAllBatch</summary>
    public static void BuildAllBatch()
    {
        int code = 0;
        try { BuildAll(); if (!BuildPlayer()) code = 1; }
        catch (System.Exception e) { Debug.LogException(e); code = 1; }
        EditorApplication.Exit(code);
    }
}
