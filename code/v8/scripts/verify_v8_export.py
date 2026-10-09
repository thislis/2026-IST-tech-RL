#!/usr/bin/env python3
"""Export a research bundle and replay it in an isolated Python process."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import argparse,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=project_root();sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-store',required=True);p.add_argument('--checkpoint-sha')
    p.add_argument('--trace',required=True);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from blackout_rl.v8.checkpoints import CheckpointStore,atomic_json
    from blackout_rl.v8.contracts import file_hash,ContractError
    from blackout_rl.v8.export import export_bundle
    from blackout_rl.v8.export_parity import replay,fixture_from_windows
    store=CheckpointStore(a.checkpoint_store);payload,sha=store.load(a.checkpoint_sha)
    bundle=export_bundle(store.directory/(sha+'.pt'),payload['config'],a.bundle).resolve()
    fixture=fixture_from_windows(a.trace);reference=replay(bundle,fixture)
    with tempfile.TemporaryDirectory(prefix='v8-clean-room-') as tmp:
        tmp=Path(tmp);atomic_json(tmp/'fixture.json',fixture)
        code='''import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from blackout_rl.v8.export_parity import replay
result=replay(sys.argv[1],json.loads(Path(sys.argv[2]).read_text()))
Path(sys.argv[3]).write_text(json.dumps(result,allow_nan=False))
'''
        command=[sys.executable,'-I','-c',code,str(bundle),str(tmp/'fixture.json'),str(tmp/'result.json')]
        subprocess.run(command,cwd=tmp,check=True,timeout=120,capture_output=True,text=True)
        isolated=json.loads((tmp/'result.json').read_text())
        if isolated['decision_sha256']!=reference['decision_sha256']:raise ContractError('export action/state parity mismatch')
        if not Path(isolated['module_path']).is_relative_to(bundle):raise ContractError('export imported project source')
    trace_files={str(a.trace):file_hash(a.trace)}
    row_store=Path(a.trace).with_name('window_rows.jsonl')
    if row_store.exists():trace_files[str(row_store)]=file_hash(row_store)
    atomic_json(a.output,dict(passed=True,checkpoint_sha256=sha,trace_sha256=file_hash(a.trace),trace_files=trace_files,bundle_manifest_sha256=file_hash(bundle/'manifest.json'),
        episodes=isolated['episodes'],requests=isolated['requests'],reference_timing=reference['timing'],isolated_timing=isolated['timing'],
        isolated_module=isolated['module_path'],decision_sha256=isolated['decision_sha256'],
        scope='isolated Python -I, empty cwd, bundle imports, installed pinned dependencies; recorded legal live observations',
        official_loader_tested=False,submission_ready=False))
    print(a.output)
if __name__=='__main__':main()
