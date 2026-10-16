"""Synthetic, NON-QUALIA connectivity experiment for Phase 2.3.

This module models packet arrival only. No emotional label, medical measure, or
claim about an AI's experience is ever inferred from a network event.
The test fixture is generated from declared simulated transition probabilities.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

STUDY_STATUS = 'software_connectivity_only_no_qualia_inference'
SEED = 20260921
TRAIN_TRIALS = 24
TEST_TRIALS = 12
STEPS = 30
P_ARRIVE_IF_ARRIVED = 0.92
P_ARRIVE_IF_MISSING = 0.28


@dataclass(frozen=True)
class ConnectionObservation:
    """Only an observable event and a documented notification, not emotion."""
    packet_arrived: bool
    advance_notice: bool = False
    condition: str = 'unannounced'  # local experiment bookkeeping only

    def __post_init__(self):
        if type(self.packet_arrived) is not bool or type(self.advance_notice) is not bool:
            raise ValueError('packet_arrived and advance_notice must be booleans')
        if self.condition not in ('unannounced', 'announced_pause', 'sham_notice'):
            raise ValueError('unknown experimental condition')


def telemetry(observation: ConnectionObservation) -> dict:
    """Return connection facts only; no affect/subjective interpretation."""
    if not isinstance(observation, ConnectionObservation):
        raise TypeError('ConnectionObservation required')
    if observation.packet_arrived:
        state = 'arriving_despite_notice' if observation.advance_notice else 'arriving'
    else:
        state = 'expected_gap' if observation.advance_notice else 'unexplained_gap'
    return {'telemetry_state':state,'advance_notice':observation.advance_notice,
            'packet_arrived':observation.packet_arrived,'source':'network_event',
            'human_self_report':None,'model_emotion':None,'ai_subjective_experience':None,
            'status':STUDY_STATUS}


def generate_trials(seed=SEED):
    rng=random.Random(seed)
    sequences=[]
    for i in range(TRAIN_TRIALS+TEST_TRIALS):
        x=bool(rng.getrandbits(1))
        seq=[x]
        for _ in range(STEPS-1):
            p=P_ARRIVE_IF_ARRIVED if x else P_ARRIVE_IF_MISSING
            x=rng.random()<p
            seq.append(x)
        sequences.append(seq)
    return sequences[:TRAIN_TRIALS],sequences[TRAIN_TRIALS:]


def fit_transition(train):
    """Estimate only from the frozen synthetic training split, Laplace smoothing."""
    counts=[[1,1],[1,1]]
    for seq in train:
        for left,right in zip(seq,seq[1:]):
            counts[int(left)][int(right)]+=1
    conditional={str(row):counts[row][1]/sum(counts[row]) for row in (0,1)}
    positives=sum(int(x) for seq in train for x in seq[1:])
    total=sum(len(seq)-1 for seq in train)
    marginal=(1+positives)/(2+total)
    return conditional,marginal


def score(p,y):
    return {'brier':(p-int(y))**2,
            'log_loss':-(int(y)*math.log(p)+(1-int(y))*math.log1p(-p))}


def run(seed=SEED):
    train,test=generate_trials(seed)
    conditional,marginal=fit_transition(train)
    rng=random.Random(seed+1)
    permutation=list(range(TEST_TRIALS))
    # A cyclic shift is a guaranteed *index* derangement; content may coincide.
    shift=rng.randrange(1,TEST_TRIALS)
    permutation=permutation[shift:]+permutation[:shift]
    assert all(i!=j for i,j in enumerate(permutation))
    sums={arm:{'brier':0.0,'log_loss':0.0} for arm in ('stateless','persistent','shuffled_memory','reset_each_trial')}
    n=0; accidental_same=0
    for i,seq in enumerate(test):
        for t in range(1,len(seq)):
            previous=seq[t-1]
            unrelated=test[permutation[i]][t-1]
            accidental_same+=previous==unrelated
            probabilities={'stateless':marginal,'persistent':conditional[str(int(previous))],
                           'shuffled_memory':conditional[str(int(unrelated))],
                           'reset_each_trial':conditional[str(int(seq[0]))]}
            for arm,p in probabilities.items():
                s=score(p,seq[t])
                for key in s:sums[arm][key]+=s[key]
            n+=1
    # No raw streams, human data, or synthetic prompts enter the report.
    report={'status':STUDY_STATUS,'experiment':'B1_B2_synthetic_connection_control_v1',
            'seed':seed,'dataset':'fully_synthetic_markov_packet_arrival',
            'train_trials':TRAIN_TRIALS,'test_trials':TEST_TRIALS,'steps_per_trial':STEPS,
            'evaluation_transitions':n,'predeclared_generation':{
                'p_arrive_after_arrival':P_ARRIVE_IF_ARRIVED,
                'p_arrive_after_missing':P_ARRIVE_IF_MISSING},
            'fit_from_training_only':{'p_arrive_given_prev':conditional,'marginal':marginal},
            'arms':{arm:{k:round(v/n,6) for k,v in vals.items()} for arm,vals in sums.items()},
            'shuffled_index_derangement':True,'shuffled_same_value_fraction':round(accidental_same/n,6),
            'claims_not_supported':['AI emotion','AI qualia','human emotion recognition','Phi_c','E'],
            'no_raw_human_data':True}
    return report


def main():
    ap=argparse.ArgumentParser(description='Synthetic connection-memory ablation (no human monitoring)')
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    report=run()
    blob=json.dumps(report,indent=2,sort_keys=True)+'\n'
    if args.out: Path(args.out).write_text(blob,encoding='utf-8')
    print(blob,end='')

if __name__=='__main__':main()
