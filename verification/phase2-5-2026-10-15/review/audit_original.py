#!/usr/bin/env python3
"""Execute the exact supplied experiments; capture omitted BCE without changing predictions."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'original/phase2_5_falsification_suite.py'
spec=importlib.util.spec_from_file_location('supplied_phase25',SOURCE)
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original
spec.loader.exec_module(original)

def summary(xs):
    m=statistics.mean(xs);sd=statistics.stdev(xs);half=1.96*sd/math.sqrt(len(xs))
    return {'mean':m,'sd':sd,'normal_mean_ci95':[m-half,m+half],'n':len(xs)}

def captured_run(function,seed):
    calls=[];brier=original.brier_score
    def capture(y,p):
        score=brier(y,p)
        calls.append({'brier':score,'bce':original.binary_cross_entropy(y,p),'transitions':len(y)})
        return score
    original.brier_score=capture
    try:returned=function(seed)
    finally:original.brier_score=brier
    return returned,calls

def seed_record(seed):
    one,calls=captured_run(original.run_experiment_1_tuned_baselines,seed)
    assert len(calls)==46
    index=round(one['oracle_tuned_ema_lr']*100)-1
    ex1={'full':calls[44],'test_selected_ema':calls[index],'fixed_reference_cusum':calls[45]}
    assert ex1['full']['brier']==one['refined_engine_v1_1_brier']
    assert ex1['test_selected_ema']['brier']==one['oracle_tuned_ema_brier']
    assert ex1['fixed_reference_cusum']['brier']==one['page_hinkley_reset_brier']
    two,calls=captured_run(original.run_experiment_2_unfamiliar_regimes,seed)
    assert len(calls)==6
    names=['overall_memory','overall_no_memory','novel_memory','novel_no_memory','revisited_memory','revisited_no_memory']
    ex2=dict(zip(names,calls))
    assert list(two.values())==[v['brier'] for v in calls]
    three,calls=captured_run(original.run_experiment_3_desynchronized_donor_control,seed)
    assert len(calls)==3
    ex3=dict(zip(['endogenous','synchronous_donor','offset_donor_as_supplied'],calls))
    assert list(three.values())==[v['brier'] for v in calls]
    return {'seed':seed,'experiment1':ex1,'experiment2':ex2,'experiment3':ex3,
            'posthoc_test_selected_lr':one['oracle_tuned_ema_lr'],
            'supplied_assertions':{
                'endogenous_beats_offset':three['endogenous_aligned_brier']<three['desynchronized_donor_brier'],
                'memory_helps_revisited':two['revisited_blocks_with_memory_brier']<two['revisited_blocks_no_memory_brier']}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=200)
    ap.add_argument('--output',type=Path,default=ROOT/'results/original_benchmark.json')
    args=ap.parse_args()
    if args.seeds<2:raise ValueError('at least two seeds required')
    rows=[seed_record(20260921+i) for i in range(args.seeds)]
    means={ex:{arm:{metric:summary([r[ex][arm][metric] for r in rows]) for metric in ('brier','bce')}
               for arm in rows[0][ex]} for ex in ('experiment1','experiment2','experiment3')}
    pairs={}
    for name,ex,a,b in [('full_minus_test_selected_ema','experiment1','full','test_selected_ema'),
                       ('full_minus_cusum','experiment1','full','fixed_reference_cusum'),
                       ('revisited_memory_minus_no_memory','experiment2','revisited_memory','revisited_no_memory'),
                       ('novel_memory_minus_no_memory','experiment2','novel_memory','novel_no_memory'),
                       ('endogenous_minus_synchronous','experiment3','endogenous','synchronous_donor'),
                       ('endogenous_minus_offset','experiment3','endogenous','offset_donor_as_supplied')]:
        pairs[name]={metric:summary([r[ex][a][metric]-r[ex][b][metric] for r in rows]) for metric in ('brier','bce')}
    failure_counts={name:sum(not r['supplied_assertions'][name] for r in rows) for name in rows[0]['supplied_assertions']}
    report={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'source_unchanged':True,'n_seeds':args.seeds,'seed_range':[20260921,20260920+args.seeds],
            'method':'Unchanged supplied experiment functions. Brier wrapper also records BCE on exactly the predictions scored by the source. EMA BCE uses the same Brier-selected rate.',
            'limitations':['EMA rate chosen on test outcomes is an explicitly hindsight-selected reference, not validation-tuned performance.',
                           'Offset donor has final B45+B23 block as supplied; no independence of schedules.',
                           'Normal intervals across seed replicates; no prospective preregistration or multiplicity correction.'],
            'single_seed':rows[0],'summary':means,'paired_differences':pairs,
            'supplied_assertion_failure_counts':failure_counts,'raw_per_seed':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for ex,arms in means.items():
        print(ex)
        for arm,v in arms.items():print(f'  {arm:28} Brier={v["brier"]["mean"]:.6f} BCE={v["bce"]["mean"]:.6f}')
    print('Failures of supplied performance assertions across seeds:',failure_counts)
    for name,v in pairs.items():print(name,*(f'{m}={d["mean"]:.6f} CI={d["normal_mean_ci95"]}' for m,d in v.items()))

if __name__=='__main__':main()
