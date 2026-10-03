"""Use the supplied game/API verbatim; no environment patches or side channels."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
RENDERED_LAUNCHER=ROOT/'launchers/v8/BlackOutRendered.app'
LAUNCHER_EXECUTABLE=RENDERED_LAUNCHER/'Contents/MacOS/BlackOutRendered'
MANIFEST=ROOT/'reports/v8/environment_restoration/restoration.json'
RETIRED_MESSAGE=(
    'v8의 수정 게임·추가 side-channel 실행 경로는 사용자 요청으로 철회되었습니다. '
    '원본 builds/BlackOut.app과 제공 blackout_env API는 보존·검증됩니다. '
    '새 모델 코드는 유지하지만 기존 v8 학습기는 추가 엔진 정보에 의존하므로 실행하지 않습니다. '
    '원본 API만 사용하는 학습·평가 연결과 별도 실험 등록이 필요합니다. '
    'Unity나 학습은 시작하지 않았습니다.'
)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify_original():
    record=json.loads(MANIFEST.read_text())
    for repo in record['upstream'].values():
        path=Path(repo['path'])
        commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
        if commit!=repo['commit'] or dirty:raise ValueError('provided source repository changed: '+str(path))
        for name,h in repo['files'].items():
            if _sha(path/name)!=h:raise ValueError('provided source changed: '+name)
    for group in ('api_files','bundle_files'):
        for name,h in record[group].items():
            if _sha(ROOT/name)!=h:raise ValueError('provided runtime changed: '+name)
    # A transport overlay could keep files intact but change the loaded API.
    import importlib.util
    for package,directory in [('blackout_env','blackout_env'),('mlagents_envs','mlagents_envs'),('google.protobuf','google/protobuf')]:
        origin=Path(importlib.util.find_spec(package).origin).resolve()
        expected=ROOT/'.venv/lib/python3.10/site-packages'/directory
        if not origin.is_relative_to(expected):raise ValueError('non-original runtime import: '+str(origin))
    return dict(state=record['status'],build=record['build'],executable_sha256=record['executable_sha256'],
                supplied_sources_unchanged=True,provided_api_unchanged=True,model_code_preserved=True,
                research_environment_enabled=False,training_ready=False,unity_started=False)


def make_original_env(*,time_scale=1.0,worker_id=0,base_port=None):
    """Return the provided class itself; reset/step/obs/reward stay upstream-owned."""
    verify_original()
    from blackout_env import BlackOutEnv
    import os
    if not LAUNCHER_EXECUTABLE.is_file() or not os.access(LAUNCHER_EXECUTABLE,os.X_OK):
        raise ValueError('rendered background launcher missing or not executable')
    # Public env_path selects an exec-only shim. No environment method, game
    # binary, reset/step, renderer or observation preprocessing is replaced.
    return BlackOutEnv(env_path=str(RENDERED_LAUNCHER),no_graphics=False,
                       time_scale=time_scale,worker_id=worker_id,base_port=base_port)


def reject_research_environment(*args,**kwargs):
    raise RuntimeError(RETIRED_MESSAGE)
