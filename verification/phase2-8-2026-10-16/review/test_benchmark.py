#!/usr/bin/env python3
"""Check experiment contracts without requiring performance superiority."""
import importlib.util
from pathlib import Path
import sys
import unittest
spec=importlib.util.spec_from_file_location('harness28',Path(__file__).with_name('benchmark_phase28.py'))
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
SELECTED={'ema':{'lr':.27},'cusum':{'lr':.18,'drift':.4,'threshold':1.}}
class HarnessTests(unittest.TestCase):
 def test_scope_masks(self):
  schedule,_=h.scenario('mixed_9',141000000);scopes,blocks=h.slices(schedule)
  expected={'overall':405,'novel':135,'white_noise':45,'novel_d':45,'novel_e':45,'familiar_return':180,'early_10':80,'postnovel_return':90}
  self.assertEqual({k:len(scopes[k]) for k in expected},expected)
  self.assertFalse(set(scopes['novel'])&set(scopes['familiar_return']))
  self.assertEqual(scopes['early_10'][:10],list(range(45,55)))
  self.assertEqual((blocks[6]['p11'],blocks[6]['p01']),(.08,.12))
 def test_stationary_noise_scope(self):
  schedule,q0=h.scenario('stationary_c',141000000);scopes,_=h.slices(schedule)
  self.assertEqual(q0,.5);self.assertEqual(len(scopes['white_noise']),405);self.assertNotIn('early_10',scopes)
 def test_baseline_grid_continuity(self):
  grid=h.candidate_grids();self.assertEqual(len(grid['ema']),50);self.assertEqual(len(grid['cusum']),384)
  self.assertIn(SELECTED['ema'],grid['ema']);self.assertIn(SELECTED['cusum'],grid['cusum'])
 def test_all_learners_use_same_fit(self):
  fit=h.p25.generate_training(141000000);models=h.models(fit,SELECTED)
  for name,obj in models.items():
   pair=(obj.predict_proba(1),obj.predict_proba(0)) if hasattr(obj,'predict_proba') else (obj.p11,obj.p01)
   self.assertEqual(pair,(fit.p11,fit.p01),name)
 def test_pinned_controls_exact_forecasts_across_eight_scenarios(self):
  for name in h.SCENARIOS:
   _,fit,stream=h.datasets(name,141000000);mods=h.models(fit,SELECTED)
   self.assertEqual(h.run(mods['restored_phase25'],stream),h.run(mods['actual_phase25'],stream),name)
   self.assertEqual(h.run(mods['no_memory'],stream),h.run(mods['actual_phase25_no_memory'],stream),name)
 def test_seed_independence_and_training_size(self):
  banks=[set(range(161000000,161000030)),set(range(141000000,141000100)),set(range(151000000,151000100))]
  for i in range(3):
   for j in range(i+1,3):self.assertFalse(banks[i]&banks[j])
  union=set.union(*banks);self.assertFalse(union&{s+1_000_000_000 for s in union})
  fit=h.p25.generate_training(141000000);self.assertEqual(fit.n0+fit.n1,696)
  self.assertEqual(h.datasets('mixed_9',141000000)[2],h.datasets('mixed_9',141000000)[2])
  self.assertNotEqual(h.datasets('mixed_9',141000000)[2],h.datasets('mixed_9',151000000)[2])
 def test_diagnostic_active_row_mapping_and_boundary(self):
  diag={'active_row':0,'step_index':47,'row_buffer_indices':[[36,41,45,47],[40,43,44,46]],
        'accepted_snapshot_index':0,'eligibility':True,'pre_retrieval_fast':(.8,.4),'after_retrieval_fast':(.8,.6),
        'surprise_trigger':True,'accepted_snapshot_probs':(.92,.85)}
  got,event=h.inspection(diag,{'index':1,'start':45},(.2,.9))
  self.assertAlmostEqual(got['active_expected_brier_change'],.3**2-.5**2)
  self.assertEqual(got['inactive_expected_brier_change'],0)
  self.assertEqual(got['accepted_with_preboundary_observation'],1)
  self.assertEqual(event['oldest_age'],11);self.assertEqual(got['row0_retrievals'],1)
 def test_scoped_event_accounting_and_direct_row_protection(self):
  result=h.evaluate(141000000,'mixed_9',SELECTED)
  events=result['scope_events'];full='phase28_full'
  self.assertEqual(events['overall'][full]['steps'],405)
  self.assertEqual(events['early_10'][full]['steps'],80)
  self.assertEqual(events['white_noise'][full]['steps'],45)
  self.assertEqual(events['novel'][full]['steps'],sum(events[s][full]['steps'] for s in ('white_noise','novel_d','novel_e')))
  for e in result['retrieval_traces'][full]:
   self.assertEqual(e['inactive_expected_brier_change'],0)
   self.assertEqual(e['row'],0 if e['pre_retrieval_fast'][0]==e['after_retrieval_fast'][0] else 1)
  self.assertEqual(events['overall'][full]['inactive_expected_brier_worsened'],0)
if __name__=='__main__':unittest.main(verbosity=2)
