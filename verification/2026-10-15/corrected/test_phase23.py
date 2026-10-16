"""Conformance to pinned independent source plus invalid-input regression tests."""
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest
import phase2_3_protocol_reproduction as fixed

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

reference=load('reference23',ROOT/'reference/repository/zora_lab/connection_study.py')
original=load('original23',ROOT/'original/phase2_3_protocol_reproduction.py')

class Phase23AuditTests(unittest.TestCase):
    def test_exact_frozen_report(self):
        frozen=json.loads((ROOT/'reference/repository/experiments/connection_study_synthetic_report.json').read_text())
        self.assertEqual(fixed.repository_report(fixed.run_single_seed()),frozen)

    def test_independent_source_conformance_200_seeds(self):
        for seed in range(20260921,20261121):
            actual=fixed.run_single_seed(fixed.Phase23Config(seed=seed))
            train,test=reference.generate_trials(seed)
            self.assertEqual(actual['train_trials'],train)
            self.assertEqual(actual['test_trials'],test)
            self.assertEqual(fixed.repository_report(actual),reference.run(seed))

    def test_stationary_extension_preserves_uploaded_metrics(self):
        a=fixed.run_single_seed(fixed.Phase23Config(mode='stationary'))
        b=original.run_single_seed()
        self.assertEqual(a['test_trials'],b['test_trials'])
        for arm in fixed.ARMS:
            for metric in ('brier','bce'):
                # log1p(-p) and log(1-p) differ by harmless accumulated roundoff.
                self.assertAlmostEqual(a['metrics'][arm][metric],b['metrics'][arm][metric],delta=1e-12)

    def test_derangement_and_first_step_reset_predictions(self):
        a=fixed.run_single_seed();n=len(a['test_trials']);sigma=a['derangement']
        self.assertEqual(sorted(sigma),list(range(n)))
        self.assertTrue(all(i!=j for i,j in enumerate(sigma)))
        same=0
        for i,seq in enumerate(a['test_trials']):
            index=i*29
            self.assertEqual(a['predictions']['stateful_markov'][index],a['predictions']['reset_each_trial'][index])
            for t in range(1,30):
                same+=seq[t-1]==a['test_trials'][sigma[i]][t-1]
        self.assertEqual(same/348,a['shuffled_same_value_fraction'])

    def test_laplace_missing_conditioning_state(self):
        model=fixed.SmoothedMarkovEstimators.fit([[1,1,1]])
        self.assertEqual(model.p01_hat,.5)
        self.assertEqual(model.p11_hat,.75)
        self.assertEqual(model.p_marginal_hat,.75)

    def test_reject_malformed_scores(self):
        for ys,ps in [([],[]),([1,0],[.5]),([2],[.5]),([1],[float('nan')]),([1],[-.1])]:
            for fn in [fixed.brier_score,fixed.binary_cross_entropy]:
                with self.assertRaises(ValueError):fn(ys,ps)
        for eps in [0,.5,float('nan')]:
            with self.assertRaises(ValueError):fixed.binary_cross_entropy([1],[.5],eps)

    def test_reject_invalid_config_and_trials(self):
        for kwargs in [dict(n_test_trials=1),dict(seq_length=1),dict(n_train_trials=0),
                       dict(laplace_alpha=0),dict(p11=float('nan')),dict(mode='bad')]:
            with self.assertRaises(ValueError):fixed.Phase23Config(**kwargs)
        for trials in [[],[[]],[[1]],[[0,1],[1]],[[0,3]]]:
            with self.assertRaises(ValueError):fixed.SmoothedMarkovEstimators.fit(trials)
        with self.assertRaises(ValueError):fixed.run_multi_seed_benchmark(1)

if __name__=='__main__':unittest.main(verbosity=2)
