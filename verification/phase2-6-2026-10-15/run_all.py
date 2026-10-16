#!/usr/bin/env python3
"""Reproduce the Phase2.6 syntax failure, repairs, audits and benchmarks."""
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
 source=ROOT/'original/phase2_6_novelty_aware_gate.py'
 assert digest(source)==p['script_sha256']
 raw=ROOT/'original/Pasted text.txt'
 if raw.exists():
  assert digest(raw)==p['upload_sha256']
  fences=re.findall(r'```python\s*\n(.*?)```',raw.read_text(),re.S)
  assert len(fences)==1 and fences[0].encode()==source.read_bytes()
 repair=ROOT/'minimal_repair/phase2_6_novelty_aware_gate.py'
 rp=json.loads((ROOT/'minimal_repair/provenance.json').read_text())
 assert digest(repair)==rp['sha256']
 assert source.read_bytes().replace(b'def _ candidate_passes_gate(',b'def _candidate_passes_gate(')==repair.read_bytes()
 ref=json.loads((ROOT/'reference/provenance.json').read_text())
 assert digest(ROOT/'reference/phase25_suite.py')==ref['sha256']
 return {'original_sha256':p['script_sha256'],'minimal_repair_sha256':rp['sha256'],'pinned_reference_sha256':ref['sha256'],'raw_upload_present_and_verified':raw.exists(),
         'corrected_engine_sha256':digest(ROOT/'corrected/phase2_6_novelty_aware_gate.py'),'harness_sha256':digest(ROOT/'review/benchmark_phase26.py')}
def main():
 before=integrity();(ROOT/'results').mkdir(exist_ok=True)
 jobs=[('original_execution.txt',['original/phase2_6_novelty_aware_gate.py'],1),
       ('minimal_repair_execution.txt',['minimal_repair/phase2_6_novelty_aware_gate.py'],0),
       ('minimal_benchmark_console.txt',['review/audit_minimal_repair.py','--output','results/minimal_benchmark.json'],0),
       ('math_check.log',['review/math_check.py','--out','results/math_check.json'],0),
       ('corrected_tests.txt',['corrected/test_phase26.py'],0),
       ('harness_tests.txt',['review/test_benchmark.py'],0),
       ('corrected_benchmark_console.txt',['review/benchmark_phase26.py','--output','results/corrected_benchmark.json'],0)]
 runs=[];counts={}
 for log,args,expected in jobs:
  print('Running',args[0],flush=True);start=time.monotonic()
  with (ROOT/'results'/log).open('w') as out:result=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
  actual=(ROOT/'results'/log).read_text().replace(str(ROOT),'[bundle]')
  (ROOT/'results'/log).write_text(actual)
  record={'command':['python3','-B',*args],'expected_exit_code':expected,'actual_exit_code':result.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'output':'results/'+log}
  if expected==1:assert 'SyntaxError' in actual,actual
  runs.append(record)
  if result.returncode!=expected:print(actual);raise SystemExit('Unexpected exit status')
  match=re.search(r'Ran (\d+) tests?',actual)
  if match:counts[args[0]]=int(match.group(1))
 after=integrity();assert before==after
 report={'status':'All expected checks reproduced; original syntax error preserved and explicitly expected','python':sys.version,'platform':platform.platform(),'integrity_before':before,'integrity_after':after,'runs':runs,'unittest_methods_by_script':counts,'unittest_methods_total':sum(counts.values()),'log_paths_normalized':'Absolute bundle prefix replaced by [bundle]; no diagnostic text otherwise changed'}
 (ROOT/'results/reproducibility_run.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Complete:',sum(counts.values()),'passing unittest methods; original SyntaxError confirmed.',flush=True)
 if (ROOT/'review/build_report.py').exists():subprocess.run([sys.executable,'-B','review/build_report.py'],cwd=ROOT,check=True)
if __name__=='__main__':main()
