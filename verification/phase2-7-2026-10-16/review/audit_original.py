#!/usr/bin/env python3
"""Run the unchanged Phase2.7 proposal; retain both scores and actual controls."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supplied27',ROOT/'original/phase2_7_kl_antirollback_gate.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def ci(xs):
 mean=statistics.mean(xs);sd=statistics.stdev(xs);half=1.96*sd/math.sqrt(len(xs))
 return {'mean':mean,'sd':sd,'n_seeds':len(xs),'ci95':[mean-half,mean+half]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/original_benchmark.json');a=ap.parse_args()
 schedule=m.mixed_familiar_novel_schedule()
 scopes={'overall':list(range(405)),'novel':[i for b in (4,5,6) for i in range(45*b,45*(b+1))],
         'familiar_return':[i for b in (2,3,7,8) for i in range(45*b,45*(b+1))],
         'early_10':[i for b in range(1,9) for i in range(45*b,45*b+10)]}
 rows=[]
 for split,base in [('test',20260921),('followup',20261921)]:
  for seed in range(base,base+100):
   stream=m.generate_scheduled_stream(schedule,seed)
   models={'phase27_full':m.Phase27Engine(),
           'proposal_ungated_mislabeled_phase25':m.Phase27Engine(mode='phase25_ungated'),
           'no_memory':m.Phase27Engine(mode='no_memory'),
           'fixed_ema_mislabeled_tuned':m.FixedEMABaseline(.16),
           'fixed_cusum_mislabeled_tuned':m.TunedCUSUMBaseline(.14,.36,1.45)}
   paths={name:[model.step_update(x,y) for x,y in zip(stream[:-1],stream[1:])] for name,model in models.items()}
   metrics={scope:{name:{'brier':m.brier_score([stream[i+1] for i in ids],[p[i] for i in ids]),
                         'bce':m.binary_cross_entropy([stream[i+1] for i in ids],[p[i] for i in ids])} for name,p in paths.items()} for scope,ids in scopes.items()}
   events={name:{key:getattr(model,key) for key in ('retrievals_accepted','retrievals_rejected_by_kl_or_rollback')} for name,model in models.items() if isinstance(model,m.Phase27Engine)}
   rows.append({'split':split,'seed':seed,'metrics':metrics,'events':events,
                'identical_overall_brier_within_source_tolerance':abs(metrics['overall']['phase27_full']['brier']-metrics['overall']['proposal_ungated_mislabeled_phase25']['brier'])<1e-12,
                'identical_predictions':paths['phase27_full']==paths['proposal_ungated_mislabeled_phase25']})
 summary={}
 for split in ('test','followup'):
  rr=[r for r in rows if r['split']==split];summary[split]={}
  for scope in scopes:
   metrics={name:{metric:ci([r['metrics'][scope][name][metric] for r in rr]) for metric in ('brier','bce')} for name in models}
   paired={name:{metric:ci([r['metrics'][scope]['phase27_full'][metric]-r['metrics'][scope][name][metric] for r in rr]) for metric in ('brier','bce')} for name in models if name!='phase27_full'}
   summary[split][scope]={'metrics':metrics,'paired_full_minus':paired}
 expected=m.evaluate_phase27_suite(100,20260921)
 names={'phase27_kl_gated':'phase27_full','phase25_ungated':'proposal_ungated_mislabeled_phase25','no_memory_baseline':'no_memory','val_tuned_ema':'fixed_ema_mislabeled_tuned','val_tuned_cusum':'fixed_cusum_mislabeled_tuned'}
 fields={'overall_brier':('overall','brier'),'overall_bce':('overall','bce'),'familiar_brier':('familiar_return','brier'),'novel_brier':('novel','brier'),'first10_brier':('early_10','brier'),'first10_bce':('early_10','bce')}
 for source_name,name in names.items():
  for field,(scope,metric) in fields.items():assert expected[source_name][field]==summary['test'][scope]['metrics'][name][metric]['mean']
 identical={split:{key:sum(r[key] for r in rows if r['split']==split) for key in ('identical_overall_brier_within_source_tolerance','identical_predictions')} for split in ('test','followup')}
 assert expected['phase27_kl_gated']['identical_runs_vs_ungated']==identical['test']['identical_overall_brier_within_source_tolerance']
 report={'status':'unchanged supplied source executed','baseline_parameters':{'ema_lr':.16,'cusum_lr':.14,'cusum_drift':.36,'cusum_threshold':1.45,'selection':'hard-coded; no validation selection in supplied script'},
         'test_seed_range':[20260921,20261020],'followup_seed_range':[20261921,20262020],'exact_source_aggregate_matches':31,
         'notes':['True A parameters and q0=1 retained exactly.','Within the supplied memory arms, retrieval filtering differs while both share the new consolidation policy; neither equals pinned Phase2.5.','Source identity metric uses Brier tolerance; exact forecast-array identity is separately recorded.'],
         'summary':summary,'identity':identical,'raw_per_seed':rows}
 a.output.write_text(json.dumps(report,indent=2)+'\n')
 print('Identity counts:',identical)
 for split,ss in summary.items():
  print(split)
  for scope,vals in ss.items():
   print(scope)
   for name,metrics in vals['metrics'].items():print(name,*(f'{k}={v["mean"]:.6f}' for k,v in metrics.items()))
   for name,metrics in vals['paired_full_minus'].items():print('full minus',name,metrics['brier']['mean'],metrics['brier']['ci95'])
if __name__=='__main__':main()
