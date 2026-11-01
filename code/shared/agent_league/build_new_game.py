"""Build a pinned source snapshot with Unity; never start the resulting game."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from .common import ROOT, atomic, read, sha
from .new_game_stage import UNITY_COMMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT.parent / 'blackout'))
    parser.add_argument('--editor', default=r'C:\Program Files\Unity\Hub\Editor\6000.3.8f1\Editor\Unity.exe')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source = Path(args.source).resolve()
    editor = Path(args.editor).resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UNITY_COMMIT:
        raise ValueError('The build source HEAD must be ' + UNITY_COMMIT)
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise ValueError('Source has local changes; preserve and review them before creating the pinned build')
    base = ROOT / 'logs/v12/new_game_build_20261101'
    base.mkdir(parents=True, exist_ok=True)
    workspace = ROOT / '.unity/build-source-af4c5faaf2ef'
    archive = ROOT / 'artifacts/source_snapshots/unity-af4c5faaf2ef.zip'
    helper = ROOT / 'code/shared/unity/BuildLeaguePlayer.cs'
    injection = workspace / 'Assets/Editor/BuildLeaguePlayer.cs'
    if not workspace.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise FileExistsError('Unregistered source archive already exists: ' + str(archive))
        subprocess.run(['git', '-C', str(source), 'archive', '--format=zip',
                        '--output', str(archive), UNITY_COMMIT], check=True)
        workspace.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if not (workspace / name).resolve().is_relative_to(workspace.resolve()):
                    raise ValueError('Source archive path escaped workspace')
            zipped.extractall(workspace)
        injection.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(helper, injection)
        atomic(base / 'source.json', dict(source_repository=str(source), source_commit=commit,
               source_archive=str(archive), source_archive_sha256=sha(archive),
               build_workspace=str(workspace), editor_only_injection=str(injection),
               editor_helper_sha256=sha(helper), original_repository_modified=False))
    else:
        record = read(base / 'source.json')
        if (record['source_commit'] != commit or record['source_archive_sha256'] != sha(archive)
                or record['editor_helper_sha256'] != sha(helper) or sha(helper) != sha(injection)):
            raise ValueError('Build source registration changed')
    if args.prepare_only:
        print('Prepared isolated build workspace; no Unity or game launched: ' + str(workspace))
        return
    if not editor.is_file():
        raise FileNotFoundError('Unity 6000.3.8f1 is not installed: ' + str(editor))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    attempt = base / stamp
    attempt.mkdir()
    output = ROOT / 'artifacts/builds' / ('windows-source-af4c5faaf2ef-' + stamp) / 'Blackout.exe'
    environment = os.environ.copy()
    environment.pop('ELECTRON_RUN_AS_NODE', None)
    environment['BLACKOUT_BUILD_OUTPUT'] = str(output)
    environment['BLACKOUT_BUILD_SUMMARY'] = str(attempt / 'build_summary.json')
    command = [str(editor), '-batchmode', '-nographics', '-quit', '-projectPath', str(workspace),
               '-buildTarget', 'Win64', '-executeMethod', 'BuildLeaguePlayer.Windows',
               '-logFile', str(attempt / 'Editor.log')]
    atomic(attempt / 'request.json', dict(command=command, source=read(base / 'source.json'),
           output=str(output), game_started=False, league_started=False))
    with (attempt / 'console.log').open('wb') as log:
        process = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        atomic(attempt / 'process.json', dict(pid=process.pid, editor=str(editor)))
        print('Unity build PID ' + str(process.pid) + '; log: ' + str(attempt / 'Editor.log'), flush=True)
        code = process.wait()
    atomic(attempt / 'exit.json', dict(exit_code=code, game_started=False, league_started=False))
    if code:
        raise SystemExit('Unity build exited ' + str(code) + '; see ' + str(attempt / 'Editor.log'))
    summary = read(attempt / 'build_summary.json')
    if summary['result'] != 'Succeeded' or not output.is_file():
        raise RuntimeError('Unity did not produce a successful Windows build')
    atomic(base / 'latest_build.json', dict(build_directory=str(output.parent),
           summary=str(attempt / 'build_summary.json'), source=read(base / 'source.json'),
           executable_sha256=sha(output), game_started=False, league_started=False))
    print('Build complete; player NOT started: ' + str(output))


if __name__ == '__main__':
    main()
