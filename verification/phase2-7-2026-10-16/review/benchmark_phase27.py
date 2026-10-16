#!/usr/bin/env python3
"""Matched Phase2.7/actual Phase2.5 and Phase2.6 audit with frozen validation choices."""
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
p27=module('corrected27',ROOT/'corrected/phase2_7_kl_antirollback_gate.py')
orig=module('supplied27',ROOT/'original/phase2_7_kl_antirollback_gate.py')
SCENARIOS=('mixed_9',*p25.SCENARIOS)
ENGINE_CONFIGS={
 'proposal_full':{},
 'proposal_ungated':{'mode':'phase25_ungated'},
 'retrieval_only_full':{'consolidation_mode':'phase25'},
 'retrieval_only_anti_only':{'consolidation_mode':'phase25','enable_kl_gate':False},
 'retrieval_only_kl_only':{'consolidation_mode':'phase25','enable_antirollback':False},
 'restored_phase25':{'mode':'phase25_ungated','consolidation_mode':'phase25'},
 'no_memory':{'mode':'no_memory','consolidation_mode':'phase25'},
 'legacy_slow_distance':{'legacy_slow_distance':True},
 'legacy_marginal_order':{'legacy_marginal_order':True}}

def scenario(name,seed):
 if name=='mixed_9':
  labels=('A','B','A','B','C','D','E','B','A')
  return [p25.Block(dwell,p11,p01,label) for (dwell,p11,p01),label in zip(orig.mixed_familiar_novel_schedule(),labels)],1.
 return p25.make_scenario(name,seed)

def datasets(name,seed):
 schedule,q0=scenario(name,seed);fit=p25.generate_training(seed)
 stream=p25.generate_stream(schedule,seed,q0=q0)
 return schedule,fit,stream

def baseline(kind,params,fit):
 model=orig.FixedEMABaseline(**params) if kind=='ema' else orig.TunedCUSUMBaseline(**params)
 model.p11,model.p01=fit.p11,fit.p01
 return model

def run(model,stream):return [model.step_update(x,y) for x,y in zip(stream[:-1],stream[1:])]

def candidate_grids():
 return {'ema':[{'lr':i/100} for i in range(1,51)],
        'cusum':[{'lr':lr,'drift':drift,'threshold':threshold}
                 for lr in (.02,.05,.10,.14,.18,.20,.35,.50)
                 for drift in (.30,.32,.36,.38,.40,.45)
                 for threshold in (.50,1.,1.25,1.45,1.65,2.5,4.,1e9)]}

def tune(n=30,base=121000000):
 data=[(s,*datasets('mixed_9',s)[1:]) for s in range(base,base+n)]
 grids=candidate_grids()
 candidates={}
 for kind,grid in grids.items():
  candidates[kind]=[]
  for params in grid:
   rows=[{'seed':seed,**p25.score(stream[1:],run(baseline(kind,params,fit),stream))} for seed,fit,stream in data]
   candidates[kind].append({'params':params,'mean_brier':statistics.mean(x['brier'] for x in rows),'mean_bce':statistics.mean(x['bce'] for x in rows),'per_seed':rows})
 selected={k:min(v,key=lambda row:row['mean_brier'])['params'] for k,v in candidates.items()}
 return {'selection':'Mean Brier only on mixed_9; stable grid-order tie breaking','seed_range':[base,base+n-1],'selected':selected,'candidate_results':candidates,
         'grid_note':'50 EMA candidates and 384 CUSUM candidates; includes the supplied fixed settings, earlier search candidates and earlier selected CUSUM parameters. Original unclipped EMA/CUSUM families retained with learned initial estimates.'}

def models(fit,selected):
 out={}
 for name,kwargs in ENGINE_CONFIGS.items():
  model=p27.Phase27Engine(**kwargs);model.initialize(fit.p11,fit.p01,fit.marginal);out[name]=model
 for name,kwargs in [('actual_phase25',{}),('actual_phase25_no_memory',{'enable_episodic_memory':False})]:
  model=p25.RefinedQualiaEngineV1_1(**kwargs);model.initialize(fit.p11,fit.p01,fit.marginal);out[name]=model
 model=p26.NoveltyAwareEngineV1_2();model.initialize(fit.p11,fit.p01,fit.marginal);out['actual_phase26']=model
 for kind in ('ema','cusum'):out['validation_'+kind]=baseline(kind,selected[kind],fit)
 out['prior_fixed_ema']=baseline('ema',{'lr':.27},fit)
 out['prior_fixed_cusum']=baseline('cusum',{'lr':.18,'drift':.40,'threshold':1.},fit)
 out['frozen_learned']=p25.PlainEMA(fit.p11,fit.p01,0.)
 return out

def slices(schedule):
 scopes={'overall':[],'familiar_return':[],'novel':[],'postnovel_return':[],'early_10':[],'later_than_10':[],'initial_30':list(range(30))}
 start=0;seen=set();novel_seen=False;blocks=[]
 for ix,b in enumerate(schedule):
  stop=start+b.dwell;ids=list(range(start,stop));scopes['overall']+=ids
  if b.label in ('C','D','E'):scopes['novel']+=ids;novel_seen=True
  if b.label in seen:
   scopes['familiar_return']+=ids
   if novel_seen and b.label in ('A','B'):scopes['postnovel_return']+=ids
  if ix:scopes['early_10']+=list(range(start,min(start+10,stop)))
  scopes['later_than_10']+=list(range(min(start+10,stop),stop))
  blocks.append({'index':ix,**asdict(b),'start':start,'end':stop});seen.add(b.label);start=stop
 return {k:v for k,v in scopes.items() if v},blocks

def evaluate(seed,name,selected):
 schedule,fit,stream=datasets(name,seed);mods=models(fit,selected)
 paths={};counters={}
 for arm,model in mods.items():
  path=[];trace=[]
  for prev,y in zip(stream[:-1],stream[1:]):
   if hasattr(model,'predict_proba'):before=model.predict_proba(prev)
   else:before=model.p11 if prev else model.p01
   value=model.step_update(prev,y);assert value==before
   path.append(value)
   if hasattr(model,'stats'):trace.append(dict(model.stats))
  paths[arm]=path
  if trace:counters[arm]=trace
 assert paths['restored_phase25']==paths['actual_phase25'], 'P25 restore prediction mismatch'
 assert paths['no_memory']==paths['actual_phase25_no_memory'], 'no-memory prediction mismatch'
 identities={'restored_phase25_equals_actual':True,'no_memory_equals_actual_phase25_no_memory':True,
  'proposal_equals_proposal_ungated':paths['proposal_full']==paths['proposal_ungated'],
  'retrieval_only_equals_actual_phase25':paths['retrieval_only_full']==paths['actual_phase25'],
  'proposal_equals_no_memory':paths['proposal_full']==paths['no_memory'],
  'retrieval_only_equals_no_memory':paths['retrieval_only_full']==paths['no_memory']}
 paths['oracle_regime']=[p11 if stream[t] else p01 for t,(p11,p01) in enumerate(p25.flatten_schedule(schedule))]
 scopes,blocks=slices(schedule);y=stream[1:]
 metrics={scope:{arm:p25.score([y[i] for i in ids],[preds[i] for i in ids]) for arm,preds in paths.items()} for scope,ids in scopes.items()}
 for block in blocks:
  start,stop=block['start'],block['end']
  block['metrics']={arm:p25.score(y[start:stop],preds[start:stop]) for arm,preds in paths.items()}
  block['events']={}
  for arm,trace in counters.items():
   before=trace[start-1] if start else dict.fromkeys(trace[0],0)
   block['events'][arm]={key:trace[stop-1][key]-before[key] for key in before}
 return {'seed':seed,'scenario':name,'training':asdict(fit),'metrics':metrics,'blocks':blocks,
         'events':{arm:trace[-1] for arm,trace in counters.items()},'prediction_identities':identities}

def interval(xs):
 mean=statistics.mean(xs);sd=statistics.stdev(xs);half=1.96*sd/math.sqrt(len(xs))
 return {'n_seeds':len(xs),'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}

def summarize(rows):
 out={}
 for split in ('test','followup'):
  out[split]={}
  for name in SCENARIOS:
   rr=[r for r in rows if r['split']==split and r['scenario']==name]
   if not rr:continue
   scopes={}
   for scope,arms in rr[0]['metrics'].items():
    scopes[scope]={'metrics':{arm:{metric:interval([r['metrics'][scope][arm][metric] for r in rr]) for metric in ('brier','bce')} for arm in arms},
                  'paired_full_minus':{arm:{metric:interval([r['metrics'][scope]['proposal_full'][metric]-r['metrics'][scope][arm][metric] for r in rr]) for metric in ('brier','bce')} for arm in arms if arm!='proposal_full'}}
   for subset,values in scopes.items():
    values['paired_retrieval_only_minus']={arm:{metric:interval([r['metrics'][subset]['retrieval_only_full'][metric]-r['metrics'][subset][arm][metric] for r in rr]) for metric in ('brier','bce')} for arm in rr[0]['metrics'][subset] if arm!='retrieval_only_full'}
   events={arm:{key:interval([r['events'][arm][key] for r in rr]) for key in rr[0]['events'][arm]} for arm in rr[0]['events']}
   out[split][name]={'scopes':scopes,'events':events,'identity_counts':{k:sum(r['prediction_identities'][k] for r in rr) for k in rr[0]['prediction_identities']}}
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/corrected_benchmark.json')
 ap.add_argument('--test-seeds',type=int,default=100);ap.add_argument('--followup-seeds',type=int,default=100);ap.add_argument('--validation-seeds',type=int,default=30)
 args=ap.parse_args()
 for n in (args.test_seeds,args.followup_seeds,args.validation_seeds):
  if n<2 or n>10000:raise ValueError('seed counts must be 2..10000')
 args.output.parent.mkdir(exist_ok=True,parents=True)
 frozen_paths=[ROOT/'corrected/phase2_7_kl_antirollback_gate.py',Path(__file__),ROOT/'reference/phase25_suite.py',ROOT/'reference/phase26_engine.py',ROOT/'original/phase2_7_kl_antirollback_gate.py']
 frozen_hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths]
 manifest=tune(args.validation_seeds)
 manifest['plan_before_test']={'scenarios':SCENARIOS,'test_range':[101000000,101000000+args.test_seeds-1],'followup_range':[111000000,111000000+args.followup_seeds-1],
  'engine_configs':ENGINE_CONFIGS,'training':'24 stationary-A trials x29 transitions, independent learned Beta(1,1) fit',
  'engine_sha256':frozen_hashes[0],
  'harness_sha256':frozen_hashes[1],
  'prior_suite_sha256':frozen_hashes[2], 'prior_phase26_sha256':frozen_hashes[3], 'original_sha256':frozen_hashes[4],
  'status':'Exploratory verification, fresh evaluation seeds; no preregistration or external replication',
  'criteria':['nonvacuous retrieval vs exact forecast identity','novel loss vs no-memory','early10 loss vs no-memory with familiar-return advantage retained']}
 manifest_path=args.output.with_name('validation_manifest.json');raw=(json.dumps(manifest,indent=2)+'\n').encode();manifest_path.write_bytes(raw)
 selected=json.loads(manifest_path.read_text())['selected']
 print('Frozen before test:',selected,hashlib.sha256(raw).hexdigest(),flush=True)
 rows=[]
 for split,base,n in [('test',101000000,args.test_seeds),('followup',111000000,args.followup_seeds)]:
  for name in SCENARIOS:
   for seed in range(base,base+n):rows.append({'split':split,**evaluate(seed,name,selected)})
   print('Completed',split,name,n,flush=True)
 report={'metadata':{'validation_manifest_sha256':hashlib.sha256(raw).hexdigest(),'selected':selected,'test_seeds':args.test_seeds,'followup_seeds':args.followup_seeds,
  'same_data':'All arms receive identical learned fits and observations. Actual Phase2.5 and Phase2.6 imported unchanged from pinned e9df7d0. Exact per-prediction equality with actual P25 and no-memory P25 asserted on every test path. Earlier selected P26 .27 EMA/.18-.40-1 CUSUM retained as fixed controls.',
  'prior_scenarios':'Six earlier generators unchanged, plus supplied nine-block mixed schedule. No donor arms in this targeted retrieval audit.',
  'statistics':'Pointwise normal95% Monte Carlo intervals from independent seed replicates; paired full minus comparator. Not adjusted for multiple comparisons, no prespecified equivalence margin.',
  'scope':'Functional statistical prediction only; no phenomenology or physical-theory confirmation'},'summary':summarize(rows),'raw_per_seed':rows}
 assert frozen_hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths], 'implementation changed during run'
 args.output.write_text(json.dumps(report,indent=2)+'\n')
 for split,ss in report['summary'].items():
  for name,vals in ss.items():
   print('\n',split,name)
   for arm,metrics in vals['scopes']['overall']['metrics'].items():print(arm,*(f'{k}={v["mean"]:.6f}' for k,v in metrics.items()))
   for scope in ('novel','familiar_return','early_10'):
    if scope not in vals['scopes']:continue
    print(scope)
    for arm in ('proposal_ungated','retrieval_only_full','actual_phase25','actual_phase26','no_memory','validation_ema','validation_cusum'):
     v=vals['scopes'][scope]['paired_full_minus'][arm]
     print('proposal minus',arm,*(f'{metric}={x["mean"]:+.6f} CI={x["ci95"]}' for metric,x in v.items()))
if __name__=='__main__':main()
