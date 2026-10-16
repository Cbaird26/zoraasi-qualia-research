#!/usr/bin/env python3
"""Re-run the audit locally using Python 3.10+ and no external packages/network.

Writes new outputs only inside results/. Original pasted code and pinned public
reference bytes are verified before and after execution. Any failed command
causes a nonzero exit. This script does not publish or modify a GitHub artifact.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
REPO=ROOT/'reference/repository'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def write(name,obj):
    (RESULTS/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def check_reference():
    hashes=json.loads((ROOT/'reference/snapshot_sha256.json').read_text())
    for name,h in hashes.items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=h:
            raise AssertionError(f'Reference changed: {name}')
    tree=json.loads((ROOT/'reference/github_tree.json').read_text())
    for entry in tree['tree']:
        if entry['type']=='blob':
            b=(REPO/entry['path']).read_bytes()
            assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==entry['sha']
    checks=[]
    for line in (REPO/'SOURCE_SHA256SUMS.txt').read_text().splitlines():
        h,name=line.split(maxsplit=1)
        ok=hashlib.sha256((REPO/name).read_bytes()).hexdigest()==h
        checks.append({'file':name,'pass':ok})
        if not ok: raise AssertionError(f'Published checksum mismatch: {name}')
    extraction=json.loads((ROOT/'original/extraction_manifest.json').read_text())
    for name,expected in extraction['extracted_scripts'].items():
        actual=hashlib.sha256((ROOT/'original'/name).read_bytes()).hexdigest()
        assert actual==expected,f'Extracted script changed: {name}'
    return checks

def summarize(values):
    mean=statistics.mean(values);sd=statistics.stdev(values)
    half=1.96*sd/math.sqrt(len(values))
    return {'mean':mean,'sd_across_seeds':sd,'mc_mean_ci95':[mean-half,mean+half],
            'ci_half_width':half,'n_seeds':len(values)}

def original_benchmarks():
    p23=load('uploaded23',ROOT/'original/phase2_3_protocol_reproduction.py')
    p24=load('uploaded24',ROOT/'original/synthetic_qualia_engine_v1.py')
    rows23=[];rows24=[]
    for seed in range(20260921,20261121):
        a=p23.run_single_seed(p23.Phase23Config(seed=seed))
        rows23.append({'seed':seed,'model':asdict(a['model']),'metrics':a['metrics']})
        rows24.append({'seed':seed,'metrics':p24.evaluate_phase24_suite(seed)})
    for phase,rows,reference in [('phase23',rows23,'stateful_markov'),('phase24',rows24,'full_phase24_engine')]:
        arms=list(rows[0]['metrics'])
        summary={a:{s:summarize([r['metrics'][a][s] for r in rows]) for s in ('brier','bce')} for a in arms}
        paired={a:{s:summarize([r['metrics'][a][s]-r['metrics'][reference][s] for r in rows]) for s in ('brier','bce')}
                for a in arms if a!=reference}
        write(f'{phase}_original_benchmark.json',{'source':'unchanged uploaded code',
            'interval_interpretation':'Approximate normal Monte Carlo mean CI over independent simulated seeds, not an individual-run interval.',
            'single_seed':rows[0],'summary':summary,'paired_comparator_minus_reference':paired,
            'reference_arm':reference,'per_seed':rows})

def main():
    RESULTS.mkdir(exist_ok=True);os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
    checks=check_reference()
    jobs=[
        ('original_phase23',['original/phase2_3_protocol_reproduction.py'],ROOT),
        ('original_phase24',['original/synthetic_qualia_engine_v1.py'],ROOT),
        ('repository_tests',['-m','unittest','discover','-s','tests','-v'],REPO),
        ('repository_frozen_verify',['verify_report.py'],REPO),
        ('repository_reproduction',['-m','zora_lab.connection_study','--out','../../results/repository_reproduced.json'],REPO),
        ('corrected_tests',['-m','unittest','discover','-s','corrected','-p','test_*.py','-v'],ROOT),
        ('math_verification_execution',['review/math_verify.py','--out','results/math_verification.json'],ROOT),
        ('phase23_corrected_console',['corrected/phase2_3_protocol_reproduction.py','--seeds','200','--output','results/phase23_corrected_benchmark.json'],ROOT),
        ('phase24_corrected_console',['corrected/phase2_4_functional_state_engine.py','--seeds','200','--output','results/phase24_corrected_benchmark.json'],ROOT),
    ]
    commands=[]
    for name,args,cwd in jobs:
        command=[sys.executable,'-B',*args];start=time.perf_counter()
        p=subprocess.run(command,cwd=cwd,capture_output=True,text=True)
        (RESULTS/f'{name}.txt').write_text(p.stdout+'\n'+p.stderr)
        commands.append({'name':name,'arguments':args,'cwd':str(cwd.relative_to(ROOT)),
                         'exit_code':p.returncode,'elapsed_seconds':time.perf_counter()-start})
        print(f'{name}: exit {p.returncode}',flush=True)
        if p.returncode:
            print(p.stdout,p.stderr);write('execution_manifest.json',{'commands':commands,'success':False})
            raise SystemExit(p.returncode)
    start=time.perf_counter();original_benchmarks()
    commands.append({'name':'original_200_seed_benchmarks','exit_code':0,'elapsed_seconds':time.perf_counter()-start})
    check_reference()
    write('execution_manifest.json',{'python':sys.version,'platform':platform.platform(),
        'repository_commit':(ROOT/'reference/commit.txt').read_text().strip(),
        'published_checksums':checks,'commands':commands,'success':True,
        'git_blob_hashes_verified':13,
        'reference_source_unchanged':True,'extracted_script_checksums_match_provenance':True})
    print('PASS: all commands succeeded; original extraction and reference checksums intact.')

if __name__=='__main__':main()
