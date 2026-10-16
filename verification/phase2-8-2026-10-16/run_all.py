#!/usr/bin/env python3
"""Reproduce the supplied Phase 2.8 script, repairs, audits and benchmarks."""
from pathlib import Path
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def integrity():
 p=json.loads((ROOT/'original/provenance.json').read_text())
 source=ROOT/'original/phase2_8_row_factored_epistemic_gate.py'
 assert digest(source)==p['script_sha256']
 raw=ROOT/'original/Pasted text.txt'
 if raw.exists():
  assert digest(raw)==p['upload_sha256']
  fences=re.findall(r'```python\s*\n(.*?)```',raw.read_text(),re.S)
  assert len(fences)==1 and fences[0].encode()==source.read_bytes()
 refs=json.loads((ROOT/'reference/provenance.json').read_text())
 for name,ref in refs.items():assert digest(ROOT/'reference'/name)==ref['sha256']
 return {'original_sha256':p['script_sha256'],
         'pinned_reference_sha256':{name:ref['sha256'] for name,ref in refs.items()},
         'raw_upload_present_and_verified':raw.exists(),
         'corrected_engine_sha256':digest(ROOT/'corrected/phase2_8_row_factored_epistemic_gate.py'),
         'harness_sha256':digest(ROOT/'review/benchmark_phase28.py')}
def main():
 before=integrity();(ROOT/'results').mkdir(exist_ok=True)
 jobs=[('original_execution.txt',['original/phase2_8_row_factored_epistemic_gate.py']),
       ('original_benchmark_console.txt',['review/audit_original.py','--output','results/original_benchmark.json']),
       ('math_check.log',['review/math_check.py','--out','results/math_check.json']),
       ('corrected_tests.txt',['corrected/test_phase28.py']),
       ('harness_tests.txt',['review/test_benchmark.py']),
       ('corrected_benchmark_console.txt',['review/benchmark_phase28.py','--output','results/corrected_benchmark.json'])]
 runs=[];counts={}
 for log,args in jobs:
  print('Running',args[0],flush=True);start=time.monotonic()
  with (ROOT/'results'/log).open('w') as out:result=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
  actual=(ROOT/'results'/log).read_text().replace(str(ROOT),'[bundle]')
  (ROOT/'results'/log).write_text(actual)
  runs.append({'command':['python3','-B',*args],'expected_exit_code':0,'actual_exit_code':result.returncode,
               'elapsed_seconds':round(time.monotonic()-start,3),'output':'results/'+log})
  if result.returncode!=0:print(actual);raise SystemExit('Unexpected exit status')
  match=re.search(r'Ran (\d+) tests?',actual)
  if match:counts[args[0]]=int(match.group(1))
 after=integrity();assert before==after
 report={'status':'All scripts, tests and benchmarks executed successfully','python':sys.version,'platform':platform.platform(),
         'integrity_before':before,'integrity_after':after,'runs':runs,'unittest_methods_by_script':counts,
         'unittest_methods_total':sum(counts.values()),
         'log_paths_normalized':'Absolute bundle prefix replaced by [bundle]; no diagnostic text otherwise changed'}
 (ROOT/'results/reproducibility_run.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Complete:',sum(counts.values()),'passing unittest methods.',flush=True)
 if (ROOT/'review/build_report.py').exists():subprocess.run([sys.executable,'-B','review/build_report.py'],cwd=ROOT,check=True)
if __name__=='__main__':main()
