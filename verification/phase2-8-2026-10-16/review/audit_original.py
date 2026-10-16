#!/usr/bin/env python3
"""Reproduce the unchanged supplied two-bank experiment and inspect actual events."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original28',ROOT/'original/phase2_8_row_factored_epistemic_gate.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def interval(xs):
 avg=statistics.mean(xs);sd=statistics.stdev(xs);half=1.96*sd/math.sqrt(len(xs))
 return {'mean':avg,'sd':sd,'n_seeds':len(xs),'ci95':[avg-half,avg+half]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/original_benchmark.json');args=ap.parse_args()
 scopes={'overall':list(range(405)),'novel':list(range(180,315)),
         'white_noise':list(range(180,225)),'novel_d':list(range(225,270)),'novel_e':list(range(270,315)),
         'familiar_return':[i for b in (2,3,7,8) for i in range(45*b,45*(b+1))],
         'early_10':[i for b in range(1,9) for i in range(45*b,45*b+10)]}
 modes=['phase28_epistemic_row','phase25_ungated','no_memory'];rows=[]
 for split,base in [('test',20260921),('followup',20261021)]:
  for seed in range(base,base+100):
   stream=m.generate_scheduled_stream(m.mixed_familiar_novel_schedule(),seed);paths={};events={}
   for mode in modes:
    model=m.Phase28Engine(mode=mode);preds=[];accepted=[];rejected=[];buffer_indices=[[],[]];mixed=[]
    for t,(prev,y) in enumerate(zip(stream[:-1],stream[1:])):
     before=(model.row_retrievals,model.rejected_candidates);preds.append(model.step_update(prev,y))
     buffer_indices[prev].append(t);buffer_indices[prev]=buffer_indices[prev][-model.min_row_obs:]
     accepted.append(model.row_retrievals-before[0]);rejected.append(model.rejected_candidates-before[1])
     mixed.append(int(accepted[-1]>0 and min(buffer_indices[prev])<45*(t//45)))
    paths[mode]=preds
    events[mode]={scope:{'accepted_retrievals':sum(accepted[i] for i in ids),
                         'rejected_checks':sum(rejected[i] for i in ids),
                         'accepted_with_preboundary_row_observation':sum(mixed[i] for i in ids)} for scope,ids in scopes.items()}
   scores={scope:{mode:{'brier':m.brier_score([stream[i+1] for i in ids],[ps[i] for i in ids]),
                       'bce':m.binary_cross_entropy([stream[i+1] for i in ids],[ps[i] for i in ids])} for mode,ps in paths.items()} for scope,ids in scopes.items()}
   rows.append({'seed':seed,'split':split,'metrics':scores,'events':events,
                'full_equals_ungated':paths[modes[0]]==paths[modes[1]],'full_equals_no_memory':paths[modes[0]]==paths[modes[2]]})
 summary={};matches=0
 fields={'overall_brier':('overall','brier'),'overall_bce':('overall','bce'),'novel_brier':('novel','brier'),'novel_bce':('novel','bce'),
         'familiar_brier':('familiar_return','brier'),'familiar_bce':('familiar_return','bce'),'block4_noise_brier':('white_noise','brier'),'first10_brier':('early_10','brier')}
 for split,base in [('test',20260921),('followup',20261021)]:
  rr=[x for x in rows if x['split']==split];ss={}
  for scope in scopes:
   ss[scope]={'metrics':{mode:{metric:interval([x['metrics'][scope][mode][metric] for x in rr]) for metric in ('brier','bce')} for mode in modes},
              'paired_full_minus':{mode:{metric:interval([x['metrics'][scope][modes[0]][metric]-x['metrics'][scope][mode][metric] for x in rr]) for metric in ('brier','bce')} for mode in modes[1:]},
              'events':{mode:{key:interval([x['events'][mode][scope][key] for x in rr]) for key in rr[0]['events'][mode][scope]} for mode in modes}}
  source=m.evaluate_phase28_suite(100,base)
  for mode in modes:
   for field,(scope,metric) in fields.items():assert source[mode][field]==ss[scope]['metrics'][mode][metric]['mean'];matches+=1
  summary[split]=ss
 result={'status':'unchanged supplied source executed','source_aggregate_exact_matches':matches,
         'seed_banks':{'test':[20260921,20261020],'followup':[20261021,20261120]},
         'notes':['Generator-truth initialization retained exactly.','Both supplied banks overlap code-development context; corrected evaluation uses fresh banks.',
                  'Source performance test runs 50 fixed seeds and is an outcome assertion, not an implementation invariant.',
                  'Original phase25_ungated uses the same storage equations but reintroduces post-outcome marginal scoring; not exact pinned Phase2.5.',
                  'Pre-boundary row-observation diagnostic describes active-row buffer; relevant to P28 only, not the ungated 6-transition rule.'],
         'summary':summary,'raw_per_seed':rows}
 args.output.write_text(json.dumps(result,indent=2)+'\n')
 print('Exact source aggregate matches:',matches)
 for split,ss in summary.items():
  print(split)
  for scope,v in ss.items():
   print(scope,{arm:{metric:round(val['mean'],6) for metric,val in vv.items()} for arm,vv in v['metrics'].items()})
   print('full minus no-memory',v['paired_full_minus']['no_memory'])
   print('full events',{key:v['events'][modes[0]][key]['mean'] for key in v['events'][modes[0]]})
if __name__=='__main__':main()
