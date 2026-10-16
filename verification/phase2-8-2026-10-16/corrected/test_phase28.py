#!/usr/bin/env python3
"""Deterministic formula, causality, replay and mechanism tests; no win assertions."""
import copy
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'corrected'))
import phase2_8_row_factored_epistemic_gate as p28


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


original = module('original28_test', ROOT/'original/phase2_8_row_factored_epistemic_gate.py')
p25 = module('pinned25_test28', ROOT/'reference/phase25_suite.py')

COMMON = ('p11_fast', 'p01_fast', 'p11_slow', 'p01_slow', 'p_marginal_online',
          'arousal', 'valence', 'continuity')


def retrieval_fixture(**kwargs):
    engine = p28.Phase28Engine(**kwargs)
    engine.initialize(.3, .2, .5)
    engine.arousal = .7
    engine.epistemic_surprise = .2
    engine.memory = [p28.EpisodicSnapshot(.9, .8, .5)]
    engine.row_buffers[1] = [1, 1, 1]
    engine.row_buffer_indices[1] = [0, 1, 2]
    engine.window6 = [(1, 1)]*3
    engine.stats['steps'] = 3
    return engine


class TestPhase28(unittest.TestCase):
    def test_known_metrics_and_validation(self):
        self.assertEqual(p28.brier_score([0, 1], [.25, .75]), .0625)
        self.assertAlmostEqual(p28.binary_cross_entropy([0, 1], [.25, .75]), -math.log(.75))
        self.assertEqual(p28.bernoulli_kl(.4, .4), 0.)
        self.assertEqual(p28.jeffreys_row_kl(4, 0, .9), 0.)
        self.assertAlmostEqual(p28.point_row_ll(3, 1, .75), 3*math.log(.75)+math.log(.25))
        self.assertTrue(math.isfinite(p28.bernoulli_kl(0, 1)))
        for targets, preds in (([], []), ([1], []), ([2], [.5]), ([0], [float('nan')])):
            with self.assertRaises(ValueError): p28.brier_score(targets, preds)
            with self.assertRaises(ValueError): p28.binary_cross_entropy(targets, preds)
        for args in ((-1, 0, .5), (1.2, 0, .5), (True, 0, .5), (1, 0, 2)):
            with self.assertRaises(ValueError): p28.jeffreys_row_kl(*args)
        for eps in (0, .5, float('nan')):
            with self.assertRaises(ValueError): p28.bernoulli_kl(.5, .5, eps)

    def test_surprise_expectation_is_not_squared_calibration_error(self):
        for q in (0., .2, .5, .8, 1.):
            for p in (.1, .3, .5, .7, .9):
                expected = q*((1-p)**2-p*(1-p))+(1-q)*(p*p-p*(1-p))
                self.assertAlmostEqual(expected, (q-p)*(1-2*p))
                if q == .5: self.assertAlmostEqual(expected, 2*(p-.5)**2)
        for y in (0, 1): self.assertEqual((y-.5)**2-.5*(1-.5), 0.)
        self.assertLess((.9-.7)*(1-2*.7), 0.)

    def test_surprise_uses_original_forecast_and_global_ema(self):
        engine = p28.Phase28Engine(mode='no_memory')
        engine.initialize(.3, .7, .5)
        engine.epistemic_surprise = .2
        self.assertEqual(engine.step_update(1, 1), .3)
        expected = .78*.2+.22*((1-.3)**2-.3*(1-.3))
        self.assertEqual(engine.epistemic_surprise, expected)
        p = engine.predict_proba(0)
        engine.step_update(0, 0)
        self.assertEqual(engine.epistemic_surprise, .78*expected+.22*(p*p-p*(1-p)))

    def test_directional_rule_and_actual_convex_contraction(self):
        gate = p28.Phase28Engine.passes_directional_contraction
        self.assertFalse(gate(.92, .60, .70, 3, 1, 0))
        self.assertFalse(gate(.9, .3, .3, 0, 0, 1))
        self.assertFalse(gate(.3, .3, .3, 4, 0, 1))
        self.assertTrue(gate(.9, .3, .3, 4, 0, 1))
        self.assertLess(abs(.65*.9+.35*.3-1), abs(.3-1))
        # Candidate already contracts toward this MLE but disagrees with latest y.
        self.assertLess(abs(.7-.75), abs(.4-.75))
        self.assertFalse(gate(.7, .4, .1, 3, 1, 0))

    def test_row_retrieval_changes_only_active_row_and_exposes_snapshot(self):
        engine = retrieval_fixture()
        self.assertEqual(engine.step_update(1, 1), .3)
        step = engine.last_step
        self.assertTrue(step['eligibility'])
        self.assertEqual(step['accepted_snapshot_index'], 0)
        self.assertEqual(step['accepted_snapshot_probs'], (.9, .8))
        self.assertEqual(step['after_retrieval_fast'][1], step['pre_retrieval_fast'][1])
        self.assertEqual(step['after_retrieval_fast'][0], .65*.9+.35*step['pre_retrieval_fast'][0])
        self.assertEqual(engine.row_retrievals, 1)
        engine.memory[0].p11 = .8
        self.assertEqual(step['accepted_snapshot_probs'], (.9, .8))

    def test_cross_row_write_ablation_changes_both_rows_only_at_retrieval(self):
        normal, both = retrieval_fixture(), retrieval_fixture(apply_row_only=False)
        normal.step_update(1, 1); both.step_update(1, 1)
        self.assertEqual(normal.last_step['pre_retrieval_fast'], both.last_step['pre_retrieval_fast'])
        self.assertEqual(normal.p11_fast, both.p11_fast)
        self.assertEqual(both.p01_fast, .65*.8+.35*.2)
        self.assertEqual(normal.p01_fast, .2)

    def test_flags_are_separate_and_fast_advantage_always_required(self):
        fixtures = [
            ('enable_surprise', lambda e: setattr(e, 'epistemic_surprise', -.5), 'surprise_blocked'),
            ('enable_cooldown', lambda e: e.row_cooldown.__setitem__(1, 3), 'cooldown_blocked'),
        ]
        for flag, mutate, reason in fixtures:
            on, off = retrieval_fixture(), retrieval_fixture(**{flag: False})
            mutate(on); mutate(off)
            on.step_update(1, 1); off.step_update(1, 1)
            self.assertEqual(on.last_step['first_step_rejection'], reason)
            self.assertIsNotNone(off.last_step['accepted_snapshot_index'])
        # Candidate-level checks use pure fixed inputs, isolating each condition.
        on = p28.Phase28Engine()
        off = p28.Phase28Engine(enable_slow_advantage=False)
        self.assertEqual(on._row_candidate_reason(.9, .3, .95, 4, 0, 1)[0], 'slow')
        self.assertIsNone(off._row_candidate_reason(.9, .3, .95, 4, 0, 1)[0])
        on = p28.Phase28Engine(enable_directional=False)
        off = p28.Phase28Engine(enable_directional=False, enable_kl=False)
        self.assertEqual(on._row_candidate_reason(.6, .1, .1, 4, 0, 1)[0], 'kl')
        self.assertIsNone(off._row_candidate_reason(.6, .1, .1, 4, 0, 1)[0])
        on = p28.Phase28Engine(enable_slow_advantage=False)
        off = p28.Phase28Engine(enable_slow_advantage=False, enable_directional=False)
        self.assertEqual(on._row_candidate_reason(.7, .4, .1, 3, 1, 0)[0], 'directional')
        self.assertIsNone(off._row_candidate_reason(.7, .4, .1, 3, 1, 0)[0])
        off = p28.Phase28Engine(enable_surprise=False, enable_cooldown=False,
            enable_kl=False, enable_slow_advantage=False, enable_directional=False)
        self.assertEqual(off._row_candidate_reason(.1, .9, .1, 4, 0, 1)[0], 'fast')

    def test_global_cooldown_expiry_and_active_row_independence(self):
        engine = retrieval_fixture()
        engine.step_update(1, 1)
        self.assertEqual(engine.row_cooldown, [0, 12])
        for k in range(1, 12):
            engine.step_update(0, 0)
            self.assertEqual(engine.row_cooldown[1], 12-k)
        engine.step_update(0, 0)
        self.assertEqual(engine.row_cooldown[1], 0)
        # The gate tests AFTER decrement, so a value of 1 already permits this step.
        engine = retrieval_fixture()
        engine.row_cooldown[1] = 1
        engine.step_update(1, 1)
        self.assertIsNotNone(engine.last_step['accepted_snapshot_index'])

    def test_gate_boundaries_and_exact_minimum_row_count(self):
        engine = retrieval_fixture()
        engine.row_buffers[1].pop()
        engine.row_buffer_indices[1].pop()
        engine.step_update(1, 1)
        self.assertEqual(engine.last_step['first_step_rejection'], 'insufficient_row')
        fast, cand = .3, .9
        delta = p28.point_row_ll(4, 0, cand)-p28.point_row_ll(4, 0, fast)
        engine = p28.Phase28Engine(retrieval_ll_margin=delta, enable_slow_advantage=False)
        self.assertEqual(engine._row_candidate_reason(cand, fast, .3, 4, 0, 1)[0], 'fast')
        engine = p28.Phase28Engine(row_kl_threshold=0.)
        self.assertIsNone(engine._row_candidate_reason(cand, fast, .3, 4, 0, 1)[0])

    def test_row_buffer_indices_ages_and_no_implicit_epoch_reset(self):
        engine = p28.Phase28Engine(mode='no_memory')
        calls = [(1, 1)]*4+[(0, 0)]*20+[(1, 0)]
        for prev, y in calls: engine.step_update(prev, y)
        self.assertEqual(engine.row_buffers[1], [1, 1, 1, 0])
        self.assertEqual(engine.row_buffer_indices[1], [1, 2, 3, 24])
        self.assertEqual(engine.last_step['row_buffer_ages'][1], [23, 22, 21, 0])
        old = copy.deepcopy(engine.last_step)
        engine.step_update(1, 0)
        self.assertEqual(old['row_buffer_indices'][1], [1, 2, 3, 24])

    def test_pre_outcome_marginal_scoring(self):
        engine = p28.Phase28Engine(mode='no_memory')
        engine.initialize(.8, .2, .4)
        self.assertEqual(engine.step_update(1, 1), .8)
        self.assertEqual(engine.valence, math.tanh(.6*(-math.log(.4)+math.log(.8))))
        self.assertEqual(engine.p_marginal_online, .95*.4+.05)
        legacy = p28.Phase28Engine(mode='no_memory', legacy_marginal_order=True)
        legacy.initialize(.8, .2, .4); legacy.step_update(1, 1)
        self.assertEqual(legacy.valence, math.tanh(.6*(-math.log(.43)+math.log(.8))))

    def test_prediction_is_pure_and_current_outcome_cannot_change_it(self):
        a = p28.Phase28Engine()
        a.initialize(.8, .2, .4)
        state = copy.deepcopy(a.__dict__)
        self.assertEqual(a.predict_proba(1), .8)
        self.assertEqual(a.__dict__, state)
        b = copy.deepcopy(a)
        self.assertEqual(a.step_update(1, 0), b.step_update(1, 1))
        self.assertNotEqual(a.p11_fast, b.p11_fast)

    def test_future_suffix_does_not_change_prefix_predictions_or_state(self):
        rng = random.Random(37)
        prefix = [rng.randrange(2) for _ in range(50)]
        streams = [prefix+[0]*20, prefix+[1]*20]
        outputs, states = [], []
        for stream in streams:
            engine = p28.Phase28Engine()
            preds = []
            for t, (prev, y) in enumerate(zip(stream[:-1], stream[1:])):
                preds.append(engine.step_update(prev, y))
                if t == 48: states.append(copy.deepcopy(engine.__dict__))
            outputs.append(preds[:49])
        self.assertEqual(outputs[0], outputs[1]); self.assertEqual(states[0], states[1])

    def test_reset_restores_all_learned_state_and_isolation(self):
        engine = retrieval_fixture()
        engine.step_update(1, 1)
        engine.initialize(.81, .22, .61)
        fresh = p28.Phase28Engine(); fresh.initialize(.81, .22, .61)
        self.assertEqual(engine.__dict__, fresh.__dict__)
        engine.row_buffers[1].append(1)
        engine.memory[0].p11 = .1
        self.assertEqual(fresh.row_buffers, [[], []]); self.assertEqual(fresh.memory[0].p11, .81)

    def test_neutral_initialization_and_zero_memory(self):
        self.assertEqual(p28.Phase28Engine().predict_proba(1), .5)
        no = p28.Phase28Engine(mode='no_memory')
        zero = p28.Phase28Engine(max_snapshots=0)
        for prev, y in [(1, 1), (1, 0), (0, 1)]*30:
            self.assertEqual(no.step_update(prev, y), zero.step_update(prev, y))
        for engine in (no, zero):
            engine._consolidate(.9, .3, .7)
            self.assertEqual(engine.memory, [])
            self.assertEqual(engine.stats['accepted_retrievals'], 0)
            self.assertEqual(engine.stats['consolidation_calls'], 0)

    def test_exact_source_replay_all_three_modes(self):
        for mode in ('phase28_epistemic_row', 'phase25_ungated', 'no_memory'):
            for seed in (20260921, 3, 417):
                old = original.Phase28Engine(mode=mode)
                new = p28.Phase28Engine(mode=mode, legacy_marginal_order=True)
                new.initialize(.92, .28, 7/9)
                stream = original.generate_scheduled_stream(original.mixed_familiar_novel_schedule(), seed)
                for prev, y in zip(stream[:-1], stream[1:]):
                    self.assertEqual(old.step_update(prev, y), new.step_update(prev, y))
                    for field in COMMON+('epistemic_surprise', 'row_cooldown', 'row_buffers',
                                         'window6', 'row_retrievals', 'rejected_candidates'):
                        self.assertEqual(getattr(old, field), getattr(new, field), (mode, seed, field))
                    if mode != 'no_memory':
                        self.assertEqual([asdict(s) for s in old.memory], [asdict(s) for s in new.memory])

    def test_exact_pinned_phase25_forecasts_and_predictive_states(self):
        for scenario in p25.SCENARIOS:
            for seed in (11, 29, 47):
                fit = p25.generate_training(seed)
                schedule, q0 = p25.make_scenario(scenario, seed)
                stream = p25.generate_stream(schedule, seed, q0=q0)
                for memory in (True, False):
                    old = p25.RefinedQualiaEngineV1_1(enable_episodic_memory=memory)
                    new = p28.Phase28Engine(mode='phase25_ungated' if memory else 'no_memory')
                    for engine in (old, new): engine.initialize(fit.p11, fit.p01, fit.marginal)
                    for prev, y in zip(stream[:-1], stream[1:]):
                        self.assertEqual(old.step_update(prev, y), new.step_update(prev, y))
                        for field in COMMON: self.assertEqual(getattr(old, field), getattr(new, field))
                        self.assertEqual(old.window, new.window6)
                        self.assertEqual(asdict(old.recorded_trace), asdict(new.recorded_trace))
                        self.assertEqual([asdict(s) for s in old.memory], [asdict(s) for s in new.memory])

    def test_flags_do_not_change_phase25_or_no_memory(self):
        for mode in ('phase25_ungated', 'no_memory'):
            a = p28.Phase28Engine(mode=mode)
            b = p28.Phase28Engine(mode=mode, enable_surprise=False, enable_kl=False,
                enable_directional=False, enable_cooldown=False, enable_slow_advantage=False, apply_row_only=False)
            rng = random.Random(71)
            prev = 1
            for _ in range(120):
                y = rng.randrange(2)
                self.assertEqual(a.step_update(prev, y), b.step_update(prev, y))
                for field in COMMON: self.assertEqual(getattr(a, field), getattr(b, field))
                prev = y

    def test_counter_partition_step_events_and_bounds(self):
        engine = retrieval_fixture()
        engine.memory.append(p28.EpisodicSnapshot(.1, .2, .5))
        all_events = {k: 0 for k in p28.COUNTERS}
        prior = dict(engine.stats)
        rng = random.Random(5)
        prev = 1
        for index in range(100):
            y = 1 if index == 0 else rng.randrange(2)
            engine.step_update(prev, y)
            for key, value in engine.last_step['events'].items(): all_events[key] += value
            for p in (engine.p11_fast, engine.p01_fast, engine.p11_slow, engine.p01_slow):
                self.assertTrue(0 <= p <= 1)
            prev = y
        self.assertEqual(engine.stats, {k: prior[k]+all_events[k] for k in all_events})
        self.assertEqual(engine.stats['candidate_evaluations'], sum(engine.stats[k] for k in (
            'rejected_fast', 'rejected_slow', 'rejected_kl', 'rejected_directional', 'gate_passed_candidates')))
        self.assertEqual(engine.stats['raw_advantage_candidates'], engine.stats['candidate_evaluations']-engine.stats['rejected_fast'])
        self.assertEqual(engine.stats['retrieval_eligible'], engine.stats['accepted_retrievals']+engine.stats['rejected_retrieval_steps'])

    def test_storage_is_phase25_rule_and_capacity_has_no_eviction(self):
        engine = p28.Phase28Engine(max_snapshots=1)
        engine.initialize(.9, .3, .7)
        engine._consolidate(.92, .32, .6)
        self.assertEqual(engine.memory[0].p11, .5*.9+.5*.92)
        self.assertEqual(engine.stats['memory_merges'], 1)
        state = asdict(engine.memory[0])
        engine._consolidate(.1, .9, .7)
        self.assertEqual(asdict(engine.memory[0]), state)
        self.assertEqual(engine.stats['capacity_rejections'], 1)

    def test_invalid_inputs_reject_before_any_mutation(self):
        bad_kwargs = ({'mode':'x'}, {'min_row_obs':0}, {'min_row_obs':1.5},
            {'max_snapshots':-1}, {'max_snapshots':True}, {'row_cooldown_steps':-.1},
            {'row_kl_threshold':-1}, {'retrieval_ll_margin':float('nan')},
            {'epistemic_surprise_thresh':float('inf')}, {'enable_kl':1}, {'apply_row_only':None})
        for kw in bad_kwargs:
            with self.assertRaises(ValueError): p28.Phase28Engine(**kw)
        engine = retrieval_fixture()
        state = copy.deepcopy(engine.__dict__)
        for prev, y in ((2, 0), (1, 2), (float('nan'), 0), (1., 0), (True, 0), (1, .5)):
            with self.assertRaises(ValueError): engine.step_update(prev, y)
            self.assertEqual(state, engine.__dict__)
        with self.assertRaises(ValueError): engine.initialize(.5, 2, .5)
        self.assertEqual(state, engine.__dict__)


if __name__ == '__main__':
    unittest.main(verbosity=2)
