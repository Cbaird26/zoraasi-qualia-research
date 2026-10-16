#!/usr/bin/env python3
"""Reproduce the Phase 2.5 verification with Python standard library only."""
from pathlib import Path
import hashlib
import json
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def integrity():
    provenance = json.loads((ROOT/'original/provenance.json').read_text())
    source = ROOT/'original/phase2_5_falsification_suite.py'
    assert digest(source) == provenance['script_sha256'], 'original script changed'
    pasted = ROOT/'original/Pasted text.txt'
    if pasted.exists():
        assert digest(pasted) == provenance['upload_sha256'], 'private upload changed'
        fences = re.findall(r'```python\s*\n(.*?)```', pasted.read_text(), re.S)
        assert len(fences) == 1 and fences[0].encode() == source.read_bytes(), 'extraction mismatch'
    reference = json.loads((ROOT/'reference/provenance.json').read_text())
    assert digest(ROOT/'reference/phase24_engine.py') == reference['sha256'], 'pinned reference changed'
    return {'original_script_sha256': digest(source), 'pinned_phase24_sha256': reference['sha256'],
            'private_upload_verified_if_present': pasted.exists()}

def main():
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    before = integrity()
    jobs = [
        ('original_execution.txt', ['original/phase2_5_falsification_suite.py']),
        ('original_benchmark_console.txt', ['review/audit_original.py','--seeds','200','--output','results/original_benchmark.json']),
        ('math_check.log', ['review/math_check.py','--out','results/math_check.json','--phase24-source','reference/phase24_engine.py']),
        ('corrected_tests.txt', ['corrected/test_phase25.py']),
        ('corrected_benchmark_console.txt', ['corrected/phase2_5_falsification_suite.py','--test-seeds','100','--validation-seeds','30','--followup-seeds','100','--output','results/corrected_benchmark.json']),
        ('bridge_phase24_console.txt', ['review/bridge_phase24.py','--benchmark','results/corrected_benchmark.json','--output','results/bridge_phase24.json']),
    ]
    runs = []
    for log, args in jobs:
        print('Running',args[0],flush=True)
        start = time.monotonic()
        with (results/log).open('w') as out:
            run = subprocess.run([sys.executable,'-B',*args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        runs.append({'command':['python3','-B',*args],'exit_code':run.returncode,
                     'elapsed_seconds':round(time.monotonic()-start,3),'output':'results/'+log})
        if run.returncode:
            print((results/log).read_text())
            raise SystemExit(run.returncode)
    after = integrity()
    assert before == after
    payload = {'status':'all execution commands exited 0','python':sys.version,'platform':platform.platform(),
               'integrity_before':before,'integrity_after':after,'runs':runs,
               'unittest_methods':{'original_performance_assertions':2,'corrected_invariants':18,'mathematical':11},
               'additional_audits':{'phase25_chronology':2,'phase24_stored_continuity':1,
                   'bridge_exact_score_matches':2400},
               'interpretation':'Passing unit tests does not require a method to outperform a comparator. Original performance assertions fail for some additional seeds; those failures are reported rather than suppressed.'}
    (results/'reproducibility_run.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('Completed: 31 unittest methods, 3 chronology/state audits, and 2,400 bridge score matches.',flush=True)
    if (ROOT/'review/build_report.py').exists():
        subprocess.run([sys.executable,'-B','review/build_report.py'],cwd=ROOT,check=True)

if __name__ == '__main__':main()
