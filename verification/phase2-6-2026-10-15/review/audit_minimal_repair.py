#!/usr/bin/env python3
"""Run the single-character repaired proposal without algorithm changes."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import math
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supplied26',ROOT/'minimal_repair/phase2_6_novelty_aware_gate.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def ci(xs):
 mean=statistics.mean(xs);sd=statistics.stdev(xs);half=1.96*sd/math.sqrt(len(xs))
 return {'mean':mean,'sd':sd,'n_seeds':len(xs),'ci95':[mean-half,mean+half]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/minimal_benchmark.json');a=ap.parse_args()
 rate,cp=m.tune_baselines_on_validation();schedule=m.mixed_familiar_novel_schedule()
 scopes={'overall':list(range(405)),'novel':[i for b in (4,5,6) for i in range(45*b,45*(b+1))],
         'familiar_return':[i for b in (2,3,7,8) for i in range(45*b,45*(b+1))],
         'early_10':[i for b in range(1,9) for i in range(45*b,45*b+10)]}
 rows=[]
 for split,base in [('test',20260921),('followup',20261921)]:
  for seed in range(base,base+100):
   stream=m.generate_scheduled_stream(schedule,seed)
   models={'phase26_full':m.NoveltyAwareEngineV1_2(),
           'phase26_gate_and_fallback_off_mislabeled_phase25':m.NoveltyAwareEngineV1_2(enable_novelty_gate=False,enable_ood_fallback=False),
           'no_memory':m.NoveltyAwareEngineV1_2(enable_episodic_memory=False,enable_novelty_gate=False),
           'validation_ema':m.FixedEMABaseline(rate),'validation_cusum':m.TunedCUSUMBaseline(*cp)}
   paths={name:[mod.step_update(x,y) for x,y in zip(stream[:-1],stream[1:])] for name,mod in models.items()}
   metrics={scope:{name:{'brier':m.brier_score([stream[i+1] for i in ids],[p[i] for i in ids]),
                         'bce':m.binary_cross_entropy([stream[i+1] for i in ids],[p[i] for i in ids])} for name,p in paths.items()} for scope,ids in scopes.items()}
   events={name:{key:getattr(mod,key) for key in ('familiar_retrievals','novelty_rejections','ood_fallbacks')} for name,mod in models.items() if isinstance(mod,m.NoveltyAwareEngineV1_2)}
   rows.append({'split':split,'seed':seed,'metrics':metrics,'events':events})
 summary={}
 for split in ('test','followup'):
  rr=[r for r in rows if r['split']==split];summary[split]={}
  for scope in scopes:
   metrics={name:{metric:ci([r['metrics'][scope][name][metric] for r in rr]) for metric in ('brier','bce')} for name in models}
   paired={name:{metric:ci([r['metrics'][scope]['phase26_full'][metric]-r['metrics'][scope][name][metric] for r in rr]) for metric in ('brier','bce')} for name in models if name!='phase26_full'}
   summary[split][scope]={'metrics':metrics,'paired_full_minus':paired}
 # Verify aggregate equals the supplied benchmark, not an approximation of it.
 expected=m.evaluate_phase26_benchmark(100,20260921)
 names={'phase26_novelty_gated':'phase26_full','phase25_ungated_memory':'phase26_gate_and_fallback_off_mislabeled_phase25','no_memory_adaptive':'no_memory','val_tuned_ema':'validation_ema','val_tuned_cusum':'validation_cusum'}
 fields={'overall_brier':('overall','brier'),'overall_bce':('overall','bce'),'familiar_brier':('familiar_return','brier'),'novel_brier':('novel','brier'),'first10_brier':('early_10','brier'),'first10_bce':('early_10','bce')}
 for source_name,name in names.items():
  for field,(scope,metric) in fields.items():assert expected[source_name][field]==summary['test'][scope]['metrics'][name][metric]['mean']
 report={'status':'executed after syntax-only repair; original cannot parse','selected_baselines':{'ema_rate':rate,'cusum_rate_drift_threshold':cp},'validation_seed_range':[30260921,30260950],'test_seed_range':[20260921,20261020],'followup_seed_range':[20261921,20262020],
         'exact_source_aggregate_matches':30,'notes':['Truth initialization retained exactly. Both default supplied tests pass after syntax-only repair.','The source phase25_ungated_memory arm is Phase2.6 with gate and fallback disabled, not actual Phase2.5.','BCE now reported for all four scopes on unchanged forecasts. Pointwise paired normal Monte Carlo intervals, no multiplicity adjustment.'],
         'summary':summary,'raw_per_seed':rows}
 a.output.write_text(json.dumps(report,indent=2)+'\n')
 print('Selected baseline parameters:',report['selected_baselines'])
 for split,scopes in summary.items():
  print(split)
  for scope,vals in scopes.items():
   print(scope)
   for name,metrics in vals['metrics'].items():print(name,*(f'{k}={v["mean"]:.6f}' for k,v in metrics.items()))
   for name,metrics in vals['paired_full_minus'].items():print('full minus',name,metrics['brier']['mean'],metrics['brier']['ci95'])
if __name__=='__main__':main()
