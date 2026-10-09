"""Validate audit links/JSON and re-hash original inputs; extract card indices."""

# Locate shared modules from the physical versioned layout.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents
                    if (p / "code/shared/project_paths.py").is_file())
_layout_sys.path.insert(0, str(_layout_root / "code/shared"))
from project_paths import activate as _activate_layout
_activate_layout()
from project_paths import project_root, project_path

import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=project_root()
OUT=Path(__file__).resolve().parent
REPORT=OUT.parent/'internal_result.md'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    report=REPORT.read_text()
    cards=[]
    for m in re.finditer(r'^### (C\d\d) — (.+)\n',report,re.M):
        tail=report[m.end():];stop=re.search(r'^#{2,3} ',tail,re.M)
        body=tail[:stop.start()] if stop else tail
        cards.append(dict(id=m[1],title=m[2],report_line=report[:m.start()].count('\n')+1,
                          evidence_markdown=body.strip(),commit='82ce2a1014c28b02408ef733c578a7ef367053cc'))
    decisions=[]
    for line in report.splitlines():
        if re.match(r'^\| D\d\d \|',line):
            c=[x.strip() for x in line.strip('|').split('|')]
            decisions.append(dict(id=c[0],links=c[1],decision=c[2]))
    write('root_cause_evidence.json',cards);write('v8_decision_inputs.json',decisions)
    files=list(OUT.glob('*.json'))
    for p in files:json.loads(p.read_text())
    broken=[]
    generated_here={OUT/'delivery_verification.json',OUT/'output_manifest.json'}
    for p in [REPORT,OUT/'run_inventory.md']:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            destination=(p.parent/target).resolve()
            if not destination.exists() and destination not in generated_here:
                broken.append(dict(file=p.name,target=target))
    expected={}
    for name in ['input_hashes.json','supplement_input_hashes.json','finalize_input_hashes.json']:
        for key,value in json.loads((OUT/name).read_text()).items():
            p=Path(key) if Path(key).is_absolute() else project_path(key, root=ROOT)
            # Generated audit outputs are tracked by output_manifest, not originals.
            if p.is_relative_to(OUT.parent):continue
            expected[p]=value['sha256']
    changed=[str(p) for p,h in expected.items() if not p.exists() or sha(p)!=h]
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    result=dict(report_lines=len(report.splitlines()),json_files_parsed=len(files),cards=len(cards),decisions=len(decisions),
                broken_links=broken,original_inputs_rehashed=len(expected),original_inputs_changed=changed,
                japanese_text_lines=[i for i,l in enumerate(report.splitlines(),1) if re.search('[ぁ-んァ-ン]',l)],git_status=status)
    write('delivery_verification.json',result)
    outputs={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size)
             for p in [REPORT]+sorted(OUT.iterdir()) if p.is_file() and p.name!='output_manifest.json'}
    write('output_manifest.json',outputs)
    assert all(p.exists() for p in generated_here)
    print(json.dumps(result,ensure_ascii=False))
    assert not broken and not changed and len(cards)==10 and len(decisions)==8

if __name__=='__main__':main()
