#!/usr/bin/env python3
"""Matched Phase2.6/actual Phase2.5 audit with frozen validation choices."""
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
p26=module('corrected26',ROOT/'corrected/phase2_6_novelty_aware_gate.py')
orig=module('syntax_repaired26',ROOT/'minimal_repair/phase2_6_novelty_aware_gate.py')
SCENARIOS=('mixed_9',*p25.SCENARIOS)
ENGINE_CONFIGS={
 'phase26_full':{},
 'phase26_gate_off':{'enable_novelty_gate':False},
 'phase26_fallback_off':{'enable_ood_fallback':False},
 'phase26_gate_and_fallback_off':{'enable_novelty_gate':False,'enable_ood_fallback':False},
 'phase26_no_memory':{'enable_episodic_memory':False,'enable_novelty_gate':False},
 'phase26_no_memory_no_fallback':{'enable_episodic_memory':False,'enable_novelty_gate':False,'enable_ood_fallback':False},
 'phase26_no_surprise_boost':{'enable_surprise_boost':False},
 'phase26_no_prediction_damping':{'enable_prediction_damping':False},
 'phase26_legacy_marginal_order':{'legacy_marginal_order':True}}

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

def tune(n=30,base=91000000):
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
         'grid_note':'50 EMA candidates and 384 CUSUM candidates; includes every supplied grid point and earlier selected CUSUM parameters. Original unclipped EMA/CUSUM families retained with learned initial estimates.'}

def models(fit,selected):
 out={}
 for name,kwargs in ENGINE_CONFIGS.items():
  model=p26.NoveltyAwareEngineV1_2(**kwargs);model.initialize(fit.p11,fit.p01,fit.marginal);out[name]=model
 for name,kwargs in [('actual_phase25',{}),('actual_phase25_no_memory',{'enable_episodic_memory':False})]:
  model=p25.RefinedQualiaEngineV1_1(**kwargs);model.initialize(fit.p11,fit.p01,fit.marginal);out[name]=model
 for kind in ('ema','cusum'):out['validation_'+kind]=baseline(kind,selected[kind],fit)
 out['prior_fixed_ema']=p25.PlainEMA(fit.p11,fit.p01,.32)
 out['prior_fixed_cusum']=p25.ErrorCUSUM(fit.p11,fit.p01,.20,1.,.38)
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
   if arm in ENGINE_CONFIGS:trace.append(dict(model.stats))
  paths[arm]=path
  if trace:counters[arm]=trace
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
         'events':{arm:trace[-1] for arm,trace in counters.items()}}

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
                  'paired_full_minus':{arm:{metric:interval([r['metrics'][scope]['phase26_full'][metric]-r['metrics'][scope][arm][metric] for r in rr]) for metric in ('brier','bce')} for arm in arms if arm!='phase26_full'}}
   events={arm:{key:interval([r['events'][arm][key] for r in rr]) for key in rr[0]['events'][arm]} for arm in rr[0]['events']}
   out[split][name]={'scopes':scopes,'events':events}
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'results/corrected_benchmark.json')
 ap.add_argument('--test-seeds',type=int,default=100);ap.add_argument('--followup-seeds',type=int,default=100);ap.add_argument('--validation-seeds',type=int,default=30)
 args=ap.parse_args()
 for n in (args.test_seeds,args.followup_seeds,args.validation_seeds):
  if n<2 or n>10000:raise ValueError('seed counts must be 2..10000')
 args.output.parent.mkdir(exist_ok=True,parents=True)
 frozen_paths=[ROOT/'corrected/phase2_6_novelty_aware_gate.py',Path(__file__),ROOT/'reference/phase25_suite.py']
 frozen_hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_paths]
 manifest=tune(args.validation_seeds)
 manifest['plan_before_test']={'scenarios':SCENARIOS,'test_range':[71000000,71000000+args.test_seeds-1],'followup_range':[81000000,81000000+args.followup_seeds-1],
  'engine_configs':ENGINE_CONFIGS,'training':'24 stationary-A trials x29 transitions, independent learned Beta(1,1) fit',
  'engine_sha256':frozen_hashes[0],
  'harness_sha256':frozen_hashes[1],
  'prior_suite_sha256':frozen_hashes[2],
  'status':'Exploratory verification, fresh evaluation seeds; no preregistration or external replication',
  'criteria':['novel loss vs no-memory','familiar-return loss vs no-memory and validation baselines','early10 Brier and BCE vs validation baselines']}
 manifest_path=args.output.with_name('validation_manifest.json');raw=(json.dumps(manifest,indent=2)+'\n').encode();manifest_path.write_bytes(raw)
 selected=json.loads(manifest_path.read_text())['selected']
 print('Frozen before test:',selected,hashlib.sha256(raw).hexdigest(),flush=True)
 rows=[]
 for split,base,n in [('test',71000000,args.test_seeds),('followup',81000000,args.followup_seeds)]:
  for name in SCENARIOS:
   for seed in range(base,base+n):rows.append({'split':split,**evaluate(seed,name,selected)})
   print('Completed',split,name,n,flush=True)
 report={'metadata':{'validation_manifest_sha256':hashlib.sha256(raw).hexdigest(),'selected':selected,'test_seeds':args.test_seeds,'followup_seeds':args.followup_seeds,
  'same_data':'All arms receive exactly the same learned initialization and observations per scenario/seed. Actual Phase2.5 imported from pinned commit7be77aa. Earlier selected .32 EMA/.2-1-.38 CUSUM are additional fixed comparators.',
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
    for arm in ('actual_phase25','phase26_no_memory','phase26_gate_off','validation_ema','validation_cusum'):
     v=vals['scopes'][scope]['paired_full_minus'][arm]
     print('full minus',arm,*(f'{metric}={x["mean"]:+.6f} CI={x["ci95"]}' for metric,x in v.items()))
if __name__=='__main__':main()
