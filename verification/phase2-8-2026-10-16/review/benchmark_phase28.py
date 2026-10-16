#!/usr/bin/env python3
"""Matched learned-initialization audit of Phase 2.8 with observed retrieval effects."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
p25=module('pinned25',ROOT/'reference/phase25_suite.py')
p26=module('pinned26',ROOT/'reference/phase26_engine.py')
p27=module('pinned27',ROOT/'reference/phase27_engine.py')
baselines=module('baseline_families',ROOT/'reference/phase27_supplied.py')
p28=module('corrected28',ROOT/'corrected/phase2_8_row_factored_epistemic_gate.py')
orig=module('supplied28',ROOT/'original/phase2_8_row_factored_epistemic_gate.py')
SCENARIOS=('mixed_9',*p25.SCENARIOS,'stationary_c')
ENGINE_CONFIGS={
 'phase28_full':{},'no_surprise':{'enable_surprise':False},'no_kl':{'enable_kl':False},
 'no_directional':{'enable_directional':False},'no_cooldown':{'enable_cooldown':False},
 'no_slow_advantage':{'enable_slow_advantage':False},'two_row_update':{'apply_row_only':False},
 'legacy_marginal_order':{'legacy_marginal_order':True},
 'restored_phase25':{'mode':'phase25_ungated'},'no_memory':{'mode':'no_memory'}}

def scenario(name,seed):
 if name=='mixed_9':
  return [p25.Block(n,a,b,label) for (n,a,b),label in zip(orig.mixed_familiar_novel_schedule(),('A','B','A','B','C','D','E','B','A'))],1.
 if name=='stationary_c':return [p25.Block(405,.5,.5,'C')],.5
 return p25.make_scenario(name,seed)
def datasets(name,seed):
 schedule,q0=scenario(name,seed);fit=p25.generate_training(seed)
 return schedule,fit,p25.generate_stream(schedule,seed,q0=q0)
def baseline(kind,params,fit):
 model=baselines.FixedEMABaseline(**params) if kind=='ema' else baselines.TunedCUSUMBaseline(**params)
 model.p11,model.p01=fit.p11,fit.p01;return model
def run(model,stream):return [model.step_update(x,y) for x,y in zip(stream[:-1],stream[1:])]
def candidate_grids():
 return {'ema':[{'lr':i/100} for i in range(1,51)],
         'cusum':[{'lr':lr,'drift':drift,'threshold':threshold} for lr in (.02,.05,.10,.14,.18,.20,.35,.50)
                  for drift in (.30,.32,.36,.38,.40,.45) for threshold in (.50,1.,1.25,1.45,1.65,2.5,4.,1e9)]}
def tune(n=30,base=161000000):
 data=[(s,*datasets('mixed_9',s)[1:]) for s in range(base,base+n)];candidates={}
 for kind,grid in candidate_grids().items():
  candidates[kind]=[]
  for params in grid:
   scores=[{'seed':s,**p25.score(stream[1:],run(baseline(kind,params,fit),stream))} for s,fit,stream in data]
   candidates[kind].append({'params':params,'mean_brier':statistics.mean(x['brier'] for x in scores),
                             'mean_bce':statistics.mean(x['bce'] for x in scores),'per_seed':scores})
 return {'selection':'Mean mixed_9 validation Brier only; deterministic grid-order ties','seed_range':[base,base+n-1],
         'selected':{kind:min(rows,key=lambda x:x['mean_brier'])['params'] for kind,rows in candidates.items()},
         'candidate_results':candidates,'grid_note':'Same 50 EMA / 384 CUSUM candidates and unchanged baseline classes as Phase2.7; fresh validation seeds.'}
def models(fit,selected):
 out={}
 for name,kwargs in ENGINE_CONFIGS.items():
  obj=p28.Phase28Engine(**kwargs);obj.initialize(fit.p11,fit.p01,fit.marginal);out[name]=obj
 for name,kwargs in [('actual_phase25',{}),('actual_phase25_no_memory',{'enable_episodic_memory':False})]:
  obj=p25.RefinedQualiaEngineV1_1(**kwargs);obj.initialize(fit.p11,fit.p01,fit.marginal);out[name]=obj
 obj=p26.NoveltyAwareEngineV1_2();obj.initialize(fit.p11,fit.p01,fit.marginal);out['actual_phase26']=obj
 for name,kwargs in [('phase27_proposal',{}),('phase27_retrieval_only',{'consolidation_mode':'phase25'})]:
  obj=p27.Phase27Engine(**kwargs);obj.initialize(fit.p11,fit.p01,fit.marginal);out[name]=obj
 for kind in ('ema','cusum'):out['validation_'+kind]=baseline(kind,selected[kind],fit)
 out['prior_fixed_ema']=baseline('ema',{'lr':.27},fit)
 out['prior_fixed_cusum']=baseline('cusum',{'lr':.18,'drift':.40,'threshold':1.},fit)
 out['frozen_learned']=p25.PlainEMA(fit.p11,fit.p01,0.)
 return out
def slices(schedule):
 scopes={k:[] for k in ('overall','familiar_return','novel','white_noise','novel_d','novel_e','postnovel_return','early_10','later_than_10','initial_30')}
 scopes['initial_30']=list(range(min(30,sum(b.dwell for b in schedule))))
 start=0;seen=set();novel_seen=False;blocks=[]
 for ix,b in enumerate(schedule):
  stop=start+b.dwell;ids=list(range(start,stop));scopes['overall']+=ids
  if b.label in ('C','D','E'):scopes['novel']+=ids;novel_seen=True
  if b.label=='C':scopes['white_noise']+=ids
  if b.label=='D':scopes['novel_d']+=ids
  if b.label=='E':scopes['novel_e']+=ids
  if b.label in seen:
   scopes['familiar_return']+=ids
   if novel_seen and b.label in ('A','B'):scopes['postnovel_return']+=ids
  if ix:scopes['early_10']+=list(range(start,min(start+10,stop)))
  scopes['later_than_10']+=list(range(min(start+10,stop),stop))
  blocks.append({'index':ix,**asdict(b),'start':start,'end':stop});seen.add(b.label);start=stop
 return {k:v for k,v in scopes.items() if v},blocks

def inspection(diag,block,true_pair):
 """Analysis only: no block metadata or truth is passed to any learner."""
 row=diag['active_row'];t=diag['step_index'];indices=diag['row_buffer_indices'][row]
 mixed=bool(indices and min(indices)<block['start']);accepted=diag['accepted_snapshot_index'] is not None
 eligible=diag['eligibility'];pre=diag['pre_retrieval_fast'];post=diag['after_retrieval_fast']
 # Pair order is (p11,p01), whereas conditioning-row order is 0,1.
 active=0 if row else 1;other=1-active
 active_delta=(post[active]-true_pair[active])**2-(pre[active]-true_pair[active])**2
 other_delta=(post[other]-true_pair[other])**2-(pre[other]-true_pair[other])**2
 out={'surprise_above_threshold_steps':int(diag['surprise_trigger']),
      'eligible_with_preboundary_observation':int(eligible and mixed),
      'accepted_with_preboundary_observation':int(accepted and mixed),
      'row0_retrievals':int(accepted and row==0),'row1_retrievals':int(accepted and row==1),
      'active_expected_brier_change':active_delta,'inactive_expected_brier_change':other_delta,
      'active_expected_brier_worsened':int(accepted and active_delta>1e-15),
      'inactive_expected_brier_worsened':int(accepted and other_delta>1e-15),
      'accepted_oldest_observation_age_sum':t-min(indices) if accepted else 0}
 return out,{'step':t,'block':block['index'],'offset':t-block['start'],'row':row,'preboundary_observation':mixed,
             'buffer_indices':indices,'oldest_age':t-min(indices),'true_parameters':true_pair,
             'pre_retrieval_fast':pre,'after_retrieval_fast':post,'active_expected_brier_change':active_delta,
             'inactive_expected_brier_change':other_delta,'snapshot':diag['accepted_snapshot_probs']} if accepted else None

def evaluate(seed,name,selected):
 schedule,fit,stream=datasets(name,seed);scopes,blocks=slices(schedule)
 truths=p25.flatten_schedule(schedule);step_blocks=[b for b in blocks for _ in range(b['end']-b['start'])]
 paths={};counter_traces={};diagnostic_traces={};retrieval_traces={}
 for arm,obj in models(fit,selected).items():
  path=[];trace=[];diag_trace=[];retrieval_trace=[]
  for t,(prev,y) in enumerate(zip(stream[:-1],stream[1:])):
   before=obj.predict_proba(prev) if hasattr(obj,'predict_proba') else (obj.p11 if prev else obj.p01)
   old=dict(obj.stats) if hasattr(obj,'stats') else None
   value=obj.step_update(prev,y);assert value==before;path.append(value)
   if old is not None:trace.append({k:obj.stats[k]-old[k] for k in old})
   if arm in ENGINE_CONFIGS and arm not in ('restored_phase25','no_memory'):
    diag,event=inspection(obj.last_step,step_blocks[t],truths[t]);diag_trace.append(diag)
    if event is not None:retrieval_trace.append(event)
  paths[arm]=path
  if trace:counter_traces[arm]=trace
  if diag_trace:diagnostic_traces[arm]=diag_trace
  if arm in ('phase28_full','two_row_update'):retrieval_traces[arm]=retrieval_trace
 assert paths['restored_phase25']==paths['actual_phase25'],'Pinned P25 forecast mismatch'
 assert paths['no_memory']==paths['actual_phase25_no_memory'],'Pinned no-memory forecast mismatch'
 identities={'restored_phase25_equals_actual':True,'no_memory_equals_actual_phase25_no_memory':True,
             **{'full_equals_'+a:paths['phase28_full']==p for a,p in paths.items() if a!='phase28_full'}}
 paths['oracle_regime']=[a if stream[t] else b for t,(a,b) in enumerate(truths)];y=stream[1:]
 metrics={scope:{a:p25.score([y[i] for i in ids],[p[i] for i in ids]) for a,p in paths.items()} for scope,ids in scopes.items()}
 scope_events={scope:{} for scope in scopes}
 for arm,trace in counter_traces.items():
  for scope,ids in scopes.items():
   event={k:sum(trace[i][k] for i in ids) for k in trace[0]}
   if arm in diagnostic_traces:
    dt=diagnostic_traces[arm];event.update({k:sum(dt[i][k] for i in ids) for k in dt[0]})
   scope_events[scope][arm]=event
 for b in blocks:
  start,stop=b['start'],b['end'];b['metrics']={a:p25.score(y[start:stop],p[start:stop]) for a,p in paths.items()}
 return {'seed':seed,'scenario':name,'training':asdict(fit),'metrics':metrics,'blocks':blocks,
         'scope_events':scope_events,'retrieval_traces':retrieval_traces,'prediction_identities':identities}
def interval(xs):
 avg=statistics.mean(xs);sd=statistics.stdev(xs);half=1.96*sd/math.sqrt(len(xs))
 return {'n_seeds':len(xs),'mean':avg,'sd':sd,'ci95':[avg-half,avg+half]}
def summarize(rows):
 out={}
 for split in ('test','followup'):
  out[split]={}
  for name in SCENARIOS:
   rr=[r for r in rows if r['split']==split and r['scenario']==name]
   if not rr:continue
   scopes={}
   for scope,arms in rr[0]['metrics'].items():
    scopes[scope]={'metrics':{a:{metric:interval([r['metrics'][scope][a][metric] for r in rr]) for metric in ('brier','bce')} for a in arms},
                  'paired_full_minus':{a:{metric:interval([r['metrics'][scope]['phase28_full'][metric]-r['metrics'][scope][a][metric] for r in rr]) for metric in ('brier','bce')} for a in arms if a!='phase28_full'},
                  'events':{a:{k:interval([r['scope_events'][scope][a][k] for r in rr]) for k in ev} for a,ev in rr[0]['scope_events'][scope].items()}}
   out[split][name]={'scopes':scopes,'identity_counts':{k:sum(r['prediction_identities'][k] for r in rr) for k in rr[0]['prediction_identities']}}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/corrected_benchmark.json')
 ap.add_argument('--test-seeds',type=int,default=100);ap.add_argument('--followup-seeds',type=int,default=100);ap.add_argument('--validation-seeds',type=int,default=30);args=ap.parse_args()
 for n in (args.test_seeds,args.followup_seeds,args.validation_seeds):
  if n<2 or n>10000:raise ValueError('seed counts must be 2..10000')
 args.output.parent.mkdir(parents=True,exist_ok=True)
 frozen=[ROOT/'corrected/phase2_8_row_factored_epistemic_gate.py',Path(__file__),*sorted((ROOT/'reference').glob('*.py')),ROOT/'original/phase2_8_row_factored_epistemic_gate.py']
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
 manifest=tune(args.validation_seeds)
 manifest['plan_before_test']={'scenarios':SCENARIOS,'test_range':[141000000,141000000+args.test_seeds-1],
  'followup_range':[151000000,151000000+args.followup_seeds-1],'engine_configs':ENGINE_CONFIGS,'frozen_sha256':hashes,
  'training':'24 stationary-A trials x29 transitions; independent learned Beta(1,1) fit',
  'status':'Exploratory verification with fresh seeds; internal freeze, not external preregistration',
  'criteria':['Noise-block retrieval activity and loss vs no-memory and pinned P25',
              'Direct inactive-row write protection vs whole-policy D/E losses',
              'Familiar-return retention, early-switch and overall Brier/BCE vs baselines']}
 manifest_path=args.output.with_name('validation_manifest.json');raw=(json.dumps(manifest,indent=2)+'\n').encode();manifest_path.write_bytes(raw)
 selected=json.loads(manifest_path.read_text())['selected'];print('Frozen before test:',selected,hashlib.sha256(raw).hexdigest(),flush=True)
 rows=[]
 for split,base,n in [('test',141000000,args.test_seeds),('followup',151000000,args.followup_seeds)]:
  for name in SCENARIOS:
   for seed in range(base,base+n):rows.append({'split':split,**evaluate(seed,name,selected)})
   print('Completed',split,name,n,flush=True)
 assert hashes=={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen}
 report={'metadata':{'validation_manifest_sha256':hashlib.sha256(raw).hexdigest(),'selected':selected,
  'test_seeds':args.test_seeds,'followup_seeds':args.followup_seeds,
  'same_data':'Identical learned fits/observations across arms; pinned P25/P26/P27 unchanged; complete path equivalence with P25 controls asserted.',
  'statistics':'Pointwise normal95% Monte Carlo intervals across independent seeds; paired full minus comparator; no multiplicity correction or equivalence margin.',
  'diagnostics':'Per-step counter deltas aggregated over exact scoring scopes. Truth-based risk changes isolate immediate retrieval movement relative to current regime; no truth or boundaries enter learner. They do not measure total downstream causal effects.',
  'scope':'Functional statistical simulation; no phenomenology or independent physical MQGT-SCF test'},'summary':summarize(rows),'raw_per_seed':rows}
 args.output.write_text(json.dumps(report,separators=(',',':'))+'\n')
 for split,ss in report['summary'].items():
  for name,vals in ss.items():
   print('\n',split,name)
   for a,metrics in vals['scopes']['overall']['metrics'].items():print(a,*(f'{k}={v["mean"]:.6f}' for k,v in metrics.items()))
   for scope in ('white_noise','novel','novel_d','novel_e','familiar_return','early_10'):
    if scope not in vals['scopes']:continue
    print(scope)
    for a in ('actual_phase25','actual_phase26','phase27_proposal','no_memory','validation_ema','validation_cusum','no_surprise','two_row_update'):
     print('full minus',a,*(f'{k}={v["mean"]:+.6f} CI={v["ci95"]}' for k,v in vals['scopes'][scope]['paired_full_minus'][a].items()))
if __name__=='__main__':main()
