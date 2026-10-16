#!/usr/bin/env python3
"""Check matched-data and scope integrity without requiring performance gains."""
from pathlib import Path
import importlib.util
import sys
import unittest
spec=importlib.util.spec_from_file_location('audit_harness',Path(__file__).with_name('benchmark_phase26.py'))
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
class HarnessTests(unittest.TestCase):
 def test_scope_masks_and_known_new_regime(self):
  schedule,q0=h.scenario('mixed_9',71000000);scopes,blocks=h.slices(schedule)
  self.assertEqual({k:len(scopes[k]) for k in ('overall','novel','familiar_return','early_10','postnovel_return')},
                   {'overall':405,'novel':135,'familiar_return':180,'early_10':80,'postnovel_return':90})
  self.assertFalse(set(scopes['novel'])&set(scopes['familiar_return']))
  self.assertEqual((blocks[6]['label'],blocks[6]['p11'],blocks[6]['p01']),('E',.08,.12))
  self.assertEqual(scopes['early_10'][:10],list(range(45,55)))
 def test_grids_include_source_and_prior_candidates(self):
  grid=h.candidate_grids();self.assertEqual(len(grid['ema']),50);self.assertEqual(len(grid['cusum']),384)
  self.assertTrue(all({'lr':i/100} in grid['ema'] for i in range(4,36,2)))
  for lr in (.10,.14,.18):
   for drift in (.32,.36,.40):
    for threshold in (1.25,1.45,1.65):self.assertIn(dict(lr=lr,drift=drift,threshold=threshold),grid['cusum'])
  self.assertIn(dict(lr=.2,drift=.38,threshold=1.),grid['cusum'])
 def test_actual_phase25_matches_pinned_suite_on_all_prior_scenarios(self):
  selected=h.p25.SelectedBaselines(.32,.2,1.,.38)
  for scenario in h.p25.SCENARIOS:
   seed=51000000;schedule,fit,stream=h.datasets(scenario,seed)
   model=h.p25.RefinedQualiaEngineV1_1();model.initialize(fit.p11,fit.p01,fit.marginal)
   actual=h.p25.score(stream[1:],h.run(model,stream))
   reference=h.p25.evaluate_seed(seed,scenario,selected)['metrics']['overall']['full_engine']
   self.assertEqual(actual,reference)
 def test_all_learners_receive_same_initial_estimates(self):
  fit=h.p25.generate_training(71000000)
  mods=h.models(fit,{'ema':{'lr':.3},'cusum':{'lr':.2,'drift':.38,'threshold':1.}})
  for name,model in mods.items():
   if hasattr(model,'predict_proba'):
    self.assertEqual(model.predict_proba(1),fit.p11,name);self.assertEqual(model.predict_proba(0),fit.p01,name)
   else:self.assertEqual((model.p11,model.p01),(fit.p11,fit.p01),name)
 def test_training_validation_primary_followup_rng_independence(self):
  banks=[set(range(91000000,91000030)),set(range(71000000,71000100)),set(range(81000000,81000100))]
  for i in range(len(banks)):
   for j in range(i+1,len(banks)):self.assertFalse(banks[i]&banks[j])
  union=set.union(*banks);self.assertFalse(union & {s+1_000_000_000 for s in union})
  train=h.p25.generate_training(71000000);self.assertEqual(train.n1+train.n0,696)
  _,_,stream=h.datasets('mixed_9',71000000)
  self.assertEqual(stream,h.datasets('mixed_9',71000000)[2]);self.assertNotEqual(stream,h.datasets('mixed_9',81000000)[2])
if __name__=='__main__':unittest.main(verbosity=2)
