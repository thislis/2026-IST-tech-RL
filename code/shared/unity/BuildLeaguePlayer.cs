// Copied into Assets/Editor only in the isolated build workspace.
// This compiles a player; it never enters Play Mode or launches the player.
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildLeaguePlayer
{
    [Serializable]
    private class Summary
    {
        public string unity_version;
        public string result;
        public string output;
        public string scene;
        public string backend;
        public int errors;
        public int warnings;
        public long bytes;
        public double seconds;
        public bool game_started = false;
    }

    public static void Windows()
    {
        if (Application.unityVersion != "6000.3.8f1")
            throw new InvalidOperationException("Required Unity version: 6000.3.8f1");
        string output = Environment.GetEnvironmentVariable("BLACKOUT_BUILD_OUTPUT");
        string record = Environment.GetEnvironmentVariable("BLACKOUT_BUILD_SUMMARY");
        if (string.IsNullOrEmpty(output) || string.IsNullOrEmpty(record))
            throw new InvalidOperationException("Build output and summary paths are required");
        const string scene = "Assets/Project/Runtime/Scenes/Prototype.unity";
        if (!File.Exists(scene)) throw new FileNotFoundException(scene);
        var backend = PlayerSettings.GetScriptingBackend(NamedBuildTarget.Standalone);
        if (backend != ScriptingImplementation.Mono2x)
            throw new InvalidOperationException("Expected the source project's Mono Windows backend");
        Directory.CreateDirectory(Path.GetDirectoryName(output));
        var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes = new[] { scene },
            locationPathName = output,
            target = BuildTarget.StandaloneWindows64,
            targetGroup = BuildTargetGroup.Standalone,
            subtarget = (int)StandaloneBuildSubtarget.Player,
            options = BuildOptions.None
        });
        var summary = report.summary;
        File.WriteAllText(record, JsonUtility.ToJson(new Summary
        {
            unity_version = Application.unityVersion,
            result = summary.result.ToString(),
            output = output,
            scene = scene,
            backend = backend.ToString(),
            errors = (int)summary.totalErrors,
            warnings = (int)summary.totalWarnings,
            bytes = (long)summary.totalSize,
            seconds = summary.totalTime.TotalSeconds
        }, true));
        if (summary.result != BuildResult.Succeeded)
            throw new InvalidOperationException("Windows build failed: " + summary.result);
        Debug.Log("BLACKOUT_BUILD_SUCCEEDED " + output);
    }
}
