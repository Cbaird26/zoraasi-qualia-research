#!/usr/bin/env python3
"""Audited Phase 2.3 reproduction, standard library only.

Default mode exactly follows source at e941c5fd17a49de71fcdbfbb60cb8d84e168877d:
fair getrandbits initial states and seed+1 randomized cyclic derangement.
The stationary mode preserves Gemini's alternative data generator and shift=1.
No claim about subjective experience or MQGT-SCF is tested here.
"""
from dataclasses import asdict, dataclass, replace
import argparse
import json
import math
from pathlib import Path
import random
import statistics

ARMS = ('stateful_markov', 'memoryless_marginal', 'reset_each_trial', 'deranged_memory')

@dataclass(frozen=True)
class Phase23Config:
    seed: int = 20260921
    p11: float = .92
    p01: float = .28
    n_train_trials: int = 24
    n_test_trials: int = 12
    seq_length: int = 30
    laplace_alpha: float = 1.0
    eps: float = 1e-12
    mode: str = 'repository'

    def __post_init__(self):
        if type(self.seed) is not int:
            raise ValueError('seed must be an integer')
        for name, minimum in [('n_train_trials',1),('n_test_trials',2),('seq_length',2)]:
            value = getattr(self,name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (self.p11,self.p01)):
            raise ValueError('transition probabilities must be finite and in [0,1]')
        if not math.isfinite(self.laplace_alpha) or self.laplace_alpha <= 0:
            raise ValueError('laplace_alpha must be positive and finite')
        if not 0 < self.eps < .5:
            raise ValueError('eps must be in (0,.5)')
        if self.mode not in ('repository','stationary'):
            raise ValueError('unknown mode')
        if self.mode == 'stationary' and self.p01 + 1 - self.p11 == 0:
            raise ValueError('stationary distribution is not unique')

def validate_trials(trials, minimum=1):
    if len(trials) < minimum or not trials or len(trials[0]) < 2:
        raise ValueError('insufficient trials or transitions')
    if any(len(s) != len(trials[0]) for s in trials):
        raise ValueError('trials must have equal lengths')
    if any(type(x) not in (int,bool) or x not in (0,1) for s in trials for x in s):
        raise ValueError('observations must be binary')

def generate_trials(n_trials, cfg, rng):
    trials=[]
    for _ in range(n_trials):
        if cfg.mode == 'repository':
            x=int(rng.getrandbits(1))
        else:
            x=int(rng.random() < cfg.p01/(cfg.p01+1-cfg.p11))
        seq=[x]
        for _ in range(cfg.seq_length-1):
            seq.append(int(rng.random() < (cfg.p11 if seq[-1] else cfg.p01)))
        trials.append(seq)
    return trials

def fixed_cyclic_derangement(n_trials, seed=20260921, mode='repository'):
    if type(n_trials) is not int or n_trials < 2:
        raise ValueError('derangement requires at least two trials')
    if mode not in ('repository','stationary'):
        raise ValueError('unknown mode')
    shift=random.Random(seed+1).randrange(1,n_trials) if mode == 'repository' else 1
    return [(i+shift)%n_trials for i in range(n_trials)]

@dataclass(frozen=True)
class SmoothedMarkovEstimators:
    p11_hat: float
    p01_hat: float
    p_marginal_hat: float
    n11: int
    n1_total: int
    n01: int
    n0_total: int

    @classmethod
    def fit(cls,trials,alpha=1.0):
        validate_trials(trials)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be positive and finite')
        counts=[[0,0],[0,0]]
        for seq in trials:
            for prev,nxt in zip(seq,seq[1:]): counts[prev][nxt]+=1
        n0,n1=map(sum,counts)
        return cls((counts[1][1]+alpha)/(n1+2*alpha),
                   (counts[0][1]+alpha)/(n0+2*alpha),
                   (counts[0][1]+counts[1][1]+alpha)/(n0+n1+2*alpha),
                   counts[1][1],n1,counts[0][1],n0)

    def predict_from_state(self,state):
        if type(state) not in (bool,int) or state not in (0,1):
            raise ValueError('state must be binary')
        return self.p11_hat if state else self.p01_hat

def validate_scores(y_true,y_prob):
    if not y_true or len(y_true) != len(y_prob):
        raise ValueError('scores require nonempty equal-length inputs')
    if any(type(y) not in (int,bool) or y not in (0,1) for y in y_true):
        raise ValueError('targets must be binary')
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in y_prob):
        raise ValueError('probabilities must be finite and in [0,1]')

def brier_score(y_true,y_prob):
    validate_scores(y_true,y_prob)
    return sum((p-y)**2 for y,p in zip(y_true,y_prob))/len(y_true)

def binary_cross_entropy(y_true,y_prob,eps=1e-12):
    validate_scores(y_true,y_prob)
    if not 0 < eps < .5: raise ValueError('eps must be in (0,.5)')
    ps=[min(max(p,eps),1-eps) for p in y_prob]
    return -sum(y*math.log(p)+(1-y)*math.log1p(-p) for y,p in zip(y_true,ps))/len(y_true)

def evaluate_four_arms(test_trials,model,cfg):
    validate_trials(test_trials,2)
    sigma=fixed_cyclic_derangement(len(test_trials),cfg.seed,cfg.mode)
    y=[];preds={arm:[] for arm in ARMS};same=0
    for i,seq in enumerate(test_trials):
        for t in range(1,len(seq)):
            y.append(seq[t]);previous=seq[t-1];unrelated=test_trials[sigma[i]][t-1]
            same+=previous == unrelated
            preds['stateful_markov'].append(model.predict_from_state(previous))
            preds['memoryless_marginal'].append(model.p_marginal_hat)
            preds['reset_each_trial'].append(model.predict_from_state(seq[0]))
            preds['deranged_memory'].append(model.predict_from_state(unrelated))
    return {'metrics':{arm:{'brier':brier_score(y,p),'bce':binary_cross_entropy(y,p,cfg.eps)}
                       for arm,p in preds.items()},
            'evaluation_transitions':len(y),'derangement':sigma,
            'shuffled_same_value_fraction':same/len(y),
            'targets':y,'predictions':preds}

def run_single_seed(cfg=Phase23Config()):
    rng=random.Random(cfg.seed)
    train=generate_trials(cfg.n_train_trials,cfg,rng)
    test=generate_trials(cfg.n_test_trials,cfg,rng)
    model=SmoothedMarkovEstimators.fit(train,cfg.laplace_alpha)
    result=evaluate_four_arms(test,model,cfg)
    return {'config':asdict(cfg),'model':asdict(model),'train_trials':train,'test_trials':test,**result}

def repository_report(result):
    """Serialize in the frozen report's schema; use only default protocol parameters."""
    cfg=result['config'];m=result['model']
    expected=asdict(Phase23Config(seed=cfg['seed']))
    if cfg != expected: raise ValueError('frozen schema requires default repository protocol')
    names={'stateless':'memoryless_marginal','persistent':'stateful_markov',
           'shuffled_memory':'deranged_memory','reset_each_trial':'reset_each_trial'}
    return {'status':'software_connectivity_only_no_qualia_inference',
            'experiment':'B1_B2_synthetic_connection_control_v1','seed':cfg['seed'],
            'dataset':'fully_synthetic_markov_packet_arrival','train_trials':24,
            'test_trials':12,'steps_per_trial':30,'evaluation_transitions':result['evaluation_transitions'],
            'predeclared_generation':{'p_arrive_after_arrival':.92,'p_arrive_after_missing':.28},
            'fit_from_training_only':{'p_arrive_given_prev':{'0':m['p01_hat'],'1':m['p11_hat']},
                                      'marginal':m['p_marginal_hat']},
            'arms':{name:{'brier':round(result['metrics'][arm]['brier'],6),
                          'log_loss':round(result['metrics'][arm]['bce'],6)} for name,arm in names.items()},
            'shuffled_index_derangement':True,
            'shuffled_same_value_fraction':round(result['shuffled_same_value_fraction'],6),
            'claims_not_supported':['AI emotion','AI qualia','human emotion recognition','Phi_c','E'],
            'no_raw_human_data':True}

def summarize(values):
    mean=statistics.mean(values);sd=statistics.stdev(values);half=1.96*sd/math.sqrt(len(values))
    return {'mean':mean,'sd_across_seeds':sd,'mc_mean_ci95':[mean-half,mean+half],
            'ci_half_width':half,'n_seeds':len(values)}

def run_multi_seed_benchmark(n_seeds=200,base_seed=20260921,mode='repository'):
    if type(n_seeds) is not int or n_seeds<2: raise ValueError('need at least two seeds')
    rows=[]
    for seed in range(base_seed,base_seed+n_seeds):
        out=run_single_seed(Phase23Config(seed=seed,mode=mode))
        rows.append({'seed':seed,'metrics':out['metrics'],
                     'shuffled_same_value_fraction':out['shuffled_same_value_fraction']})
    summary={a:{s:summarize([r['metrics'][a][s] for r in rows]) for s in ('brier','bce')} for a in ARMS}
    paired={a:{s:summarize([r['metrics'][a][s]-r['metrics']['stateful_markov'][s] for r in rows])
               for s in ('brier','bce')} for a in ARMS if a!='stateful_markov'}
    return {'mode':mode,'interval_interpretation':'Approximate normal 95% Monte Carlo CI for the mean over simulated independent runs; not a prediction interval or human inference.',
            'summary':summary,'paired_difference_comparator_minus_stateful':paired,'per_seed':rows}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['repository','stationary'],default='repository')
    ap.add_argument('--seeds',type=int,default=200)
    ap.add_argument('--seed',type=int,default=20260921)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();single=run_single_seed(Phase23Config(seed=args.seed,mode=args.mode))
    result={'single_seed':single,'benchmark':run_multi_seed_benchmark(args.seeds,args.seed,args.mode)}
    if args.mode=='repository':result['repository_rounded_report']=repository_report(single)
    for arm,scores in single['metrics'].items():print(f'{arm:24} Brier={scores["brier"]:.6f} BCE={scores["bce"]:.6f}')
    print('same-value fraction:',single['shuffled_same_value_fraction'],'permutation:',single['derangement'])
    for arm,metrics in result['benchmark']['summary'].items():
        print(arm, ' | '.join(f'{s}={v["mean"]:.6f} +/- {v["ci_half_width"]:.6f}' for s,v in metrics.items()))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
