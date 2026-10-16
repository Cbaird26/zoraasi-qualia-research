"""Phase 2.6 deterministic invariants; no test enforces a model ranking."""
import copy
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_6_novelty_aware_gate as m


class TestPhase26(unittest.TestCase):
    def test_integrated_row_evidence_known_values(self):
        self.assertAlmostEqual(m.beta_binomial_row_log_evidence(0, 0), 0.)
        self.assertAlmostEqual(m.beta_binomial_row_log_evidence(4, 0), math.log(1/5))
        self.assertAlmostEqual(m.beta_binomial_row_log_evidence(2, 0), math.log(1/3))
        self.assertAlmostEqual(m.beta_binomial_row_log_evidence(2, 1), math.log(1/12))
        self.assertAlmostEqual(m.point_row_log_likelihood(2, 1, .25), 2*math.log(.25)+math.log(.75))
        self.assertEqual(m.point_row_log_likelihood(0, 0, 0.), 0.)

    def test_reachable_window_rejects_mismatched_snapshot(self):
        states = [0, 1, 1, 1, 1, 1, 0, 1, 1]
        gated = m.NoveltyAwareEngineV1_2()
        gated.window = list(zip(states[:-1], states[1:]))
        counts = gated._window_counts()
        self.assertEqual(counts, (5, 1, 2, 0))
        n11, n10, n01, n00 = counts
        self.assertEqual(n01-n10, states[-1]-states[0])
        fast = m.point_row_log_likelihood(n11, n10, .15)+m.point_row_log_likelihood(n01, n00, .85)
        novel1, novel0 = m.beta_binomial_row_log_evidence(n11, n10), m.beta_binomial_row_log_evidence(n01, n00)
        snapshot = m.EpisodicSnapshot(.92, .28, .5)
        plain = m.NoveltyAwareEngineV1_2(enable_novelty_gate=False)
        self.assertTrue(plain._candidate_passes_gate(snapshot, *counts, fast, novel1, novel0)[0])
        self.assertFalse(gated._candidate_passes_gate(snapshot, *counts, fast, novel1, novel0)[0])

    def test_row_gate_requires_two_observations_and_accepts_floor_equality(self):
        engine = m.NoveltyAwareEngineV1_2(novelty_bf_threshold=-100., row_bf_floor=0.)
        snapshot = m.EpisodicSnapshot(.2, .8, .5)
        ll_one = m.point_row_log_likelihood(1, 0, .2)
        # A discordant row with just one observation is intentionally skipped.
        self.assertTrue(engine._candidate_passes_gate(snapshot, 1, 0, 0, 0, -100., ll_one+1, 0.)[0])
        ll_two = m.point_row_log_likelihood(2, 0, .2)
        self.assertFalse(engine._candidate_passes_gate(snapshot, 2, 0, 0, 0, -100., ll_two+1, 0.)[0])
        self.assertTrue(engine._candidate_passes_gate(snapshot, 2, 0, 0, 0, -100., ll_two, 0.)[0])

    def test_relative_and_global_gate_strict_thresholds(self):
        snapshot = m.EpisodicSnapshot(.5, .5, .5)
        relative = m.NoveltyAwareEngineV1_2(adv_margin=0., enable_novelty_gate=False)
        self.assertFalse(relative._candidate_passes_gate(snapshot, 0, 0, 0, 0, 0., 0., 0.)[0])
        global_gate = m.NoveltyAwareEngineV1_2(novelty_bf_threshold=0.)
        self.assertFalse(global_gate._candidate_passes_gate(snapshot, 0, 0, 0, 0, -1., 0., 0.)[0])

    def test_corrected_valence_uses_pre_observation_marginal(self):
        corrected = m.NoveltyAwareEngineV1_2()
        legacy = m.NoveltyAwareEngineV1_2(legacy_marginal_order=True)
        self.assertEqual(corrected.step_update(1, 1), .5)
        legacy.step_update(1, 1)
        self.assertEqual(corrected.valence, 0.)
        self.assertAlmostEqual(legacy.valence, math.tanh(.6*(-math.log(.53)+math.log(.5))))
        self.assertEqual(corrected.p_marginal_online, .53)

    def test_prediction_is_pure_and_returned_before_current_outcome(self):
        engine = m.NoveltyAwareEngineV1_2()
        engine.initialize(.8, .3, .6)
        before = copy.deepcopy(engine.__dict__)
        expected = engine.predict_proba(1)
        self.assertEqual(engine.__dict__, before)
        twin = copy.deepcopy(engine)
        self.assertEqual(engine.step_update(1, 0), expected)
        self.assertEqual(twin.step_update(1, 1), expected)

    def test_future_outcome_changes_do_not_affect_prefix_predictions(self):
        sequence = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
        changed = sequence[:5]+[1-y for y in sequence[5:]]
        paths = []
        for observations in (sequence, changed):
            engine = m.NoveltyAwareEngineV1_2()
            engine.initialize(.9, .2, .7)
            paths.append([engine.step_update(p, y) for p, y in zip(observations[:-1], observations[1:])])
        self.assertEqual(paths[0][:5], paths[1][:5])

    def test_full_reset_learned_initialization(self):
        fit = ((9+1)/(10+2), (2+1)/(7+2), (11+1)/(17+2))
        used, fresh = m.NoveltyAwareEngineV1_2(), m.NoveltyAwareEngineV1_2()
        for i in range(100):
            used.step_update(i % 2, (i+1) % 2)
        used.initialize(*fit)
        fresh.initialize(*fit)
        self.assertEqual(used.__dict__, fresh.__dict__)
        self.assertEqual((used.p11_fast, used.p01_fast, used.p_marginal_online), fit)
        self.assertEqual((used.memory[0].p11, used.memory[0].p01), fit[:2])
        self.assertEqual(used.window, [])
        self.assertTrue(all(value == 0 for value in used.stats.values()))

    def test_neutral_defaults_and_disabled_or_zero_capacity_memory(self):
        engine = m.NoveltyAwareEngineV1_2()
        self.assertEqual((engine.p11_fast, engine.p01_fast, engine.p_marginal_online), (.5, .5, .5))
        for config in ({"max_snapshots": 0}, {"enable_episodic_memory": False}):
            model = m.NoveltyAwareEngineV1_2(**config)
            model.initialize(.9, .2, .7)
            self.assertEqual(model.memory, [])
            model._consolidate(.9, .2, .5)
            for i in range(100):
                model.step_update(1, 1)
            self.assertEqual(model.memory, [])
            self.assertEqual(model.stats["memory_stores"], 0)
            self.assertEqual(model.stats["candidate_evaluations"], 0)

    def test_memory_instances_do_not_alias(self):
        one, two = m.NoveltyAwareEngineV1_2(), m.NoveltyAwareEngineV1_2()
        one.memory[0].p11 = .7
        one.window.append((1, 0))
        one.stats["steps"] = 5
        self.assertEqual(two.memory[0].p11, .5)
        self.assertEqual(two.window, [])
        self.assertEqual(two.stats["steps"], 0)

    @staticmethod
    def force_surprise_context(engine):
        engine.p11_fast = engine.p01_fast = .9
        engine.p11_slow = engine.p01_slow = .9
        engine.arousal = .9
        engine.continuity = .8  # no prediction damping in this fixture
        engine.window = [(1, 0)]*8

    def test_six_step_retrieval_cooldown_counts_actual_suppression(self):
        engine = m.NoveltyAwareEngineV1_2(enable_ood_fallback=False)
        engine.memory = [m.EpisodicSnapshot(.1, .1, .5)]
        self.force_surprise_context(engine)
        engine.step_update(1, 0)
        self.assertEqual(engine.familiar_retrievals, 1)
        self.assertEqual(engine.cooldown, 6)
        for remaining in (5, 4, 3, 2, 1):
            self.force_surprise_context(engine)
            engine.step_update(1, 0)
            self.assertEqual(engine.cooldown, remaining)
            self.assertEqual(engine.familiar_retrievals, 1)
        self.force_surprise_context(engine)
        engine.step_update(1, 0)
        self.assertEqual(engine.familiar_retrievals, 2)
        self.assertEqual(engine.stats["cooldown_blocked_steps"], 5)

    def test_ood_fallback_is_separate_from_damping(self):
        fallback = m.NoveltyAwareEngineV1_2(enable_episodic_memory=False, enable_prediction_damping=False)
        no_fallback = m.NoveltyAwareEngineV1_2(enable_episodic_memory=False, enable_ood_fallback=False,
                                            enable_prediction_damping=False)
        for model in (fallback, no_fallback):
            self.force_surprise_context(model)
            model.step_update(1, 0)
        self.assertEqual(fallback.ood_fallbacks, 1)
        self.assertEqual(fallback.cooldown, 5)
        self.assertEqual(no_fallback.ood_fallbacks, 0)
        self.assertEqual(fallback.stats["damping_activations"], 0)

    def test_prediction_damping_default_and_independent_flag(self):
        self.assertTrue(m.NoveltyAwareEngineV1_2().enable_prediction_damping)
        self.assertFalse(m.NoveltyAwareEngineV1_2(enable_ood_fallback=False).enable_prediction_damping)
        model = m.NoveltyAwareEngineV1_2(enable_ood_fallback=False, enable_prediction_damping=True)
        model.p11_fast, model.continuity, model.window = .9, .1, [(1, 0), (1, 0)]
        before = copy.deepcopy(model.__dict__)
        self.assertAlmostEqual(model.predict_proba(1), .65*.9 + .35*.25)
        self.assertEqual(model.__dict__, before)
        model.step_update(1, 0)
        self.assertEqual(model.stats["damping_activations"], 1)
        self.assertEqual(model.stats["ood_fallbacks"], 0)

    def test_surprise_boost_flag_removes_only_instantaneous_term(self):
        full = m.NoveltyAwareEngineV1_2(enable_episodic_memory=False, enable_ood_fallback=False)
        ablated = m.NoveltyAwareEngineV1_2(enable_episodic_memory=False, enable_ood_fallback=False,
                                         enable_surprise_boost=False)
        self.assertEqual(full.step_update(1, 1), ablated.step_update(1, 1))
        self.assertEqual(full.arousal, ablated.arousal)
        self.assertEqual(full.valence, ablated.valence)
        self.assertAlmostEqual(full.last_learning_rate-ablated.last_learning_rate, .22*.25)

    def test_candidate_counters_partition_and_aliases(self):
        engine = m.NoveltyAwareEngineV1_2()
        rng = random.Random(123)
        prev = 1
        for t in range(500):
            p11, p01 = (.92, .28) if (t//30) % 2 == 0 else (.15, .85)
            current = int(rng.random() < (p11 if prev else p01))
            engine.step_update(prev, current)
            prev = current
        rejects = sum(engine.stats[key] for key in ("rejected_advantage", "rejected_global_novelty", "rejected_row1", "rejected_row0"))
        self.assertEqual(rejects+engine.stats["gate_passed_candidates"], engine.stats["candidate_evaluations"])
        self.assertEqual(engine.familiar_retrievals, engine.stats["accepted_retrievals"])
        self.assertEqual(engine.ood_fallbacks, engine.stats["ood_fallbacks"])
        self.assertEqual(engine.novelty_rejections, engine.stats["novelty_rejection_steps"])
        before = copy.deepcopy(engine.stats)
        engine._candidate_passes_gate(m.EpisodicSnapshot(.5, .5, .5), 2, 0, 0, 0, -10., -2., 0.)
        self.assertEqual(engine.stats, before)

    def test_consolidation_capacity_and_counters(self):
        engine = m.NoveltyAwareEngineV1_2(max_snapshots=2)
        engine._consolidate(.95, .1, .5)
        engine._consolidate(.94, .11, .6)
        engine._consolidate(.1, .95, .5)
        self.assertEqual(engine.stats["memory_stores"], 1)
        self.assertEqual(engine.stats["memory_merges"], 1)
        self.assertEqual(engine.stats["capacity_rejections"], 1)
        self.assertEqual(len(engine.memory), 2)

    def test_bounds_window_limits_and_valid_extreme_initial_values(self):
        engine = m.NoveltyAwareEngineV1_2()
        engine.initialize(0., 1., 0.)
        for t in range(300):
            p = engine.step_update(t % 2, (t//3) % 2)
            self.assertTrue(math.isfinite(p) and 0 <= p <= 1)
            self.assertTrue(-1 <= engine.valence <= 1)
            for value in (engine.arousal, engine.continuity, engine.p11_fast, engine.p01_fast,
                          engine.p11_slow, engine.p01_slow, engine.p_marginal_online):
                self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
            self.assertTrue(.02 <= engine.last_learning_rate <= .42)
            self.assertLessEqual(len(engine.window), 8)
            self.assertLessEqual(len(engine.memory), 8)

    def test_legacy_equations_exactly_match_minimally_repaired_source(self):
        path = Path(__file__).resolve().parents[1] / "minimal_repair" / "phase2_6_novelty_aware_gate.py"
        spec = importlib.util.spec_from_file_location("phase26_minimal_repair", path)
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        stream = original.generate_scheduled_stream(original.mixed_familiar_novel_schedule(), 20260921)
        for config in ({}, {"enable_novelty_gate": False}, {"enable_ood_fallback": False},
                       {"enable_episodic_memory": False}):
            before = original.NoveltyAwareEngineV1_2(**config)
            after = m.NoveltyAwareEngineV1_2(legacy_marginal_order=True, **config)
            after.initialize(.92, .28, 7/9)
            for prev, y in zip(stream[:-1], stream[1:]):
                self.assertEqual(before.step_update(prev, y), after.step_update(prev, y))
                for key in ("p11_fast", "p01_fast", "p11_slow", "p01_slow", "p_marginal_online", "arousal",
                            "valence", "continuity", "cooldown", "stable_steps", "familiar_retrievals",
                            "novelty_rejections", "ood_fallbacks"):
                    self.assertEqual(getattr(before, key), getattr(after, key))

    def test_validation_rejects_before_reset_or_step_mutation(self):
        for value in (float("nan"), float("inf"), -.1, 1.1):
            engine = m.NoveltyAwareEngineV1_2()
            before = copy.deepcopy(engine.__dict__)
            with self.assertRaises(ValueError):
                engine.initialize(.9, value, .6)
            self.assertEqual(engine.__dict__, before)
        for config in ({"max_snapshots": -1}, {"window_size": 0}, {"window_size": 1.5},
                       {"adv_margin": -.1}, {"row_bf_floor": float("nan")},
                       {"enable_prediction_damping": "yes"}, {"enable_surprise_boost": 1}):
            with self.assertRaises(ValueError):
                m.NoveltyAwareEngineV1_2(**config)
        engine = m.NoveltyAwareEngineV1_2()
        for prev, y in ((2, 1), (1, 2), (1, float("nan"))):
            before = copy.deepcopy(engine.__dict__)
            with self.assertRaises(ValueError):
                engine.step_update(prev, y)
            self.assertEqual(engine.__dict__, before)
        for count in (-1, 1.2, True):
            with self.assertRaises(ValueError):
                m.beta_binomial_row_log_evidence(count, 0)
        with self.assertRaises(ValueError):
            m.point_row_log_likelihood(1, 1, .5, eps=0.)
        with self.assertRaises(ValueError):
            m.EpisodicSnapshot(.9, .2, 2.)

    def test_score_known_values_and_shape_validation(self):
        self.assertEqual(m.brier_score([0, 1], [.5, .5]), .25)
        self.assertAlmostEqual(m.binary_cross_entropy([0, 1], [.5, .5]), math.log(2))
        self.assertEqual(m.brier_score([0, 1], [0., 1.]), 0.)
        with self.assertRaises(ValueError):
            m.brier_score([0, 1], [.5])
        with self.assertRaises(ValueError):
            m.binary_cross_entropy([], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
