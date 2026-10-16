"""Phase 2.7 software/causality invariants, never stochastic model-win assertions."""
import copy
import importlib.util
import itertools
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_7_kl_antirollback_gate as m


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPhase27(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.p25 = load_module("phase27_pinned_p25", root / "reference" / "phase25_suite.py")
        cls.original = load_module("phase27_original", root / "original" / "phase2_7_kl_antirollback_gate.py")

    def test_kl_and_jeffreys_known_cases(self):
        self.assertEqual(m.bernoulli_kl(.5, .5), 0.)
        self.assertAlmostEqual(m.bernoulli_kl(.5, .25), .5*math.log(4/3))
        expected = .75*math.log(.75/.28)+.25*math.log(.25/.72)
        self.assertAlmostEqual(m.jeffreys_row_kl(1, 0, .28), expected)
        self.assertAlmostEqual(m.jeffreys_row_kl(0, 0, .5), 0.)
        self.assertAlmostEqual(m.point_row_log_likelihood(2, 1, .25), 2*math.log(.25)+math.log(.75))
        for p, q in itertools.product((.01, .2, .5, .8, .99), repeat=2):
            self.assertGreaterEqual(m.bernoulli_kl(p, q), -1e-15)

    def test_every_six_transition_binary_path_obeys_flow(self):
        engine = m.Phase27Engine()
        for path in itertools.product((0, 1), repeat=7):
            engine.window = list(zip(path[:-1], path[1:]))
            n11, n10, n01, n00 = engine._window_counts()
            self.assertEqual(n01-n10, path[-1]-path[0])
            self.assertLessEqual(abs(n01-n10), 1)
            self.assertEqual(n11+n10+n01+n00, 6)

    def test_reachable_default_six_transition_gate_example(self):
        path = [0, 1, 1, 1, 1, 1, 1]
        gated = m.Phase27Engine()
        gated.window = list(zip(path[:-1], path[1:]))
        counts = gated._window_counts()
        self.assertEqual(counts, (5, 0, 1, 0))
        self.assertEqual(len(gated.window), gated.window_size)
        snapshot = m.EpisodicSnapshot(.92, .28, .5)
        ll = 6*math.log(.5)
        ungated = m.Phase27Engine(mode="phase25_ungated")
        self.assertTrue(ungated.candidate_passes_phase27_gate(snapshot, *counts, ll, ll))
        passed, reason = gated._candidate_gate_result(snapshot, *counts, ll, ll, slow_reference=(.5, .5))
        self.assertFalse(passed)
        self.assertEqual(reason, "rejected_kl_row0")

    def test_distance_boundary_is_inclusive_and_explicit_reference_wins(self):
        engine = m.Phase27Engine(enable_kl_gate=False, anti_rollback_dist=.25)
        engine.p11_slow, engine.p01_slow = .75, .5
        snap = m.EpisodicSnapshot(.75, .5, .5)
        args = (snap, 2, 1, 1, 2, -100., -100.)
        # Proposed code rejects strictly less than radius, so equality passes.
        self.assertTrue(engine.candidate_passes_phase27_gate(*args, slow_reference=(.5, .5)))
        self.assertFalse(engine.candidate_passes_phase27_gate(*args))
        self.assertFalse(engine.candidate_passes_phase27_gate(*args, slow_reference=(.625, .5)))
        self.assertTrue(engine.candidate_passes_phase27_gate(*args, slow_reference=(.25, .5)))

    def test_slow_likelihood_and_distance_use_identical_pre_step_reference(self):
        engines = [m.Phase27Engine(legacy_slow_distance=flag) for flag in (False, True)]
        for engine in engines:
            engine.initialize(.9, .3, .5)
            engine.p11_slow, engine.p01_slow = .8, .2
            engine.arousal = .9
            engine.window = [(1, 0)]*5
            engine.step_update(1, 0)
            self.assertEqual(engine.last_gate_likelihood_reference, (.8, .2))
            self.assertEqual(engine.last_gate_ll_slow, engine._window_ll(.8, .2))
        corrected, legacy = engines
        self.assertEqual(corrected.last_gate_slow_reference, (.8, .2))
        self.assertEqual(legacy.last_gate_slow_reference, (legacy.p11_slow, legacy.p01_slow))
        self.assertNotEqual(legacy.last_gate_slow_reference, legacy.last_gate_likelihood_reference)

    def test_relative_and_kl_threshold_boundaries(self):
        snap = m.EpisodicSnapshot(.5, .5, .5)
        for likelihood in ("grouped", "sequential"):
            engine = m.Phase27Engine(mode="phase25_ungated", likelihood_mode=likelihood)
            self.assertFalse(engine.candidate_passes_phase27_gate(snap, 0, 0, 0, 0, -.3, -100.))
        threshold = m.jeffreys_row_kl(1, 0, .28)
        engine = m.Phase27Engine(enable_antirollback=False, kl_row_threshold=threshold)
        snapshot = m.EpisodicSnapshot(.75, .28, .5)
        self.assertTrue(engine.candidate_passes_phase27_gate(snapshot, 1, 0, 1, 0, -100., -100.))
        engine.kl_row_threshold = math.nextafter(threshold, -math.inf)
        self.assertFalse(engine.candidate_passes_phase27_gate(snapshot, 1, 0, 1, 0, -100., -100.))

    def test_component_flags_and_missing_rows(self):
        snap = m.EpisodicSnapshot(.5, .5, .5)
        base = m.Phase27Engine()
        no_anti = m.Phase27Engine(enable_antirollback=False)
        no_kl = m.Phase27Engine(enable_antirollback=False, enable_kl_gate=False)
        args = (snap, 3, 3, 0, 0, -100., -100.)
        self.assertEqual(base._candidate_gate_result(*args)[1], "rejected_distance")
        self.assertEqual(no_anti._candidate_gate_result(*args)[1], "rejected_missing_rows")
        self.assertTrue(no_kl.candidate_passes_phase27_gate(*args))
        self.assertFalse(m.Phase27Engine(mode="no_memory", enable_antirollback=False,
                                         enable_kl_gate=False).candidate_passes_phase27_gate(*args))

    def test_corrected_marginal_advantage_uses_pre_outcome_probabilities(self):
        corrected = m.Phase27Engine()
        legacy = m.Phase27Engine(legacy_marginal_order=True)
        self.assertEqual(corrected.step_update(1, 1), .5)
        legacy.step_update(1, 1)
        self.assertEqual(corrected.valence, 0.)
        self.assertAlmostEqual(legacy.valence, math.tanh(.6*(-math.log(.525)+math.log(.5))))
        self.assertEqual(corrected.p_marginal_online, .525)

    def test_prediction_is_pure_and_precedes_outcome(self):
        engine = m.Phase27Engine()
        engine.initialize(.999, .001, .7)
        before = copy.deepcopy(engine.__dict__)
        self.assertEqual(engine.predict_proba(1), .999)
        self.assertEqual(engine.predict_proba(0), .001)
        self.assertEqual(engine.__dict__, before)
        twin = copy.deepcopy(engine)
        self.assertEqual(engine.step_update(1, 0), .999)
        self.assertEqual(twin.step_update(1, 1), .999)

    def test_future_outcomes_do_not_change_prefix_predictions(self):
        stream = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]
        changed = stream[:5]+[1-x for x in stream[5:]]
        for consolidation in ("proposal", "phase25"):
            paths = []
            for sequence in (stream, changed):
                engine = m.Phase27Engine(consolidation_mode=consolidation)
                engine.initialize(.9, .2, .7)
                paths.append([engine.step_update(prev, y) for prev, y in zip(sequence[:-1], sequence[1:])])
            self.assertEqual(paths[0][:5], paths[1][:5])

    def test_full_reset_learned_initialization(self):
        fit = self.p25.generate_training(123)
        used, fresh = m.Phase27Engine(), m.Phase27Engine()
        for i in range(100):
            used.step_update(i % 2, (i+1) % 2)
        used.initialize(fit.p11, fit.p01, fit.marginal)
        fresh.initialize(fit.p11, fit.p01, fit.marginal)
        self.assertEqual(used.__dict__, fresh.__dict__)
        self.assertEqual((used.p11_fast, used.p01_fast, used.p_marginal_online), (fit.p11, fit.p01, fit.marginal))
        self.assertTrue(all(value == 0 for value in used.stats.values()))
        self.assertEqual(used.recorded_trace.arousal, [])

    def test_neutral_defaults_no_memory_and_zero_capacity(self):
        neutral = m.Phase27Engine()
        self.assertEqual((neutral.p11_fast, neutral.p01_fast, neutral.p_marginal_online), (.5, .5, .5))
        for options in ({"mode": "no_memory"}, {"max_snapshots": 0}):
            engine = m.Phase27Engine(**options)
            self.assertEqual(engine.memory, [])
            engine._consolidate(.9, .1, .7)
            for _ in range(100):
                engine.step_update(1, 1)
            self.assertEqual(engine.memory, [])
            self.assertEqual(engine.stats["accepted_retrievals"], 0)
            self.assertEqual(engine.stats["memory_stores"], 0)

    def test_proposal_and_phase25_storage_rules_are_explicitly_different(self):
        proposal = m.Phase27Engine(consolidation_mode="proposal")
        prior = m.Phase27Engine(consolidation_mode="phase25")
        self.assertEqual(proposal.likelihood_mode, "grouped")
        self.assertEqual(prior.likelihood_mode, "sequential")
        proposal._consolidate(.64, .5, .7)
        prior._consolidate(.64, .5, .7)
        self.assertEqual(len(proposal.memory), 1)  # .14 distance merges below .16
        self.assertEqual(len(prior.memory), 2)  # .14 distance does not merge below .12
        for engine in (proposal, prior):
            engine.initialize(.9, .1, .5)
            engine.window = [(1, 1)]*5
            engine.stable_steps = 0
            engine.valence = .7
            engine.arousal = .1
            engine.step_update(1, 1)
        self.assertEqual(proposal.stats["consolidation_calls"], 0)
        self.assertEqual(prior.stats["consolidation_calls"], 1)

    def test_mutable_memory_and_histories_are_independent(self):
        first, second = m.Phase27Engine(), m.Phase27Engine()
        first.memory[0].p11 = .9
        first.window.append((1, 0))
        first.recorded_trace.valence.append(.5)
        first.stats["steps"] = 7
        self.assertEqual(second.memory[0].p11, .5)
        self.assertEqual(second.window, [])
        self.assertEqual(second.recorded_trace.valence, [])
        self.assertEqual(second.stats["steps"], 0)

    def test_exact_pinned_phase25_forecasts_and_predictive_state(self):
        common = ("p11_fast", "p01_fast", "p11_slow", "p01_slow", "p_marginal_online", "arousal", "valence", "continuity", "window")
        for scenario in self.p25.SCENARIOS:
            for seed in (123, 20260921, 51000000):
                fit = self.p25.generate_training(seed)
                schedule, q0 = self.p25.make_scenario(scenario, seed)
                stream = self.p25.generate_stream(schedule, seed, q0=q0)
                for enabled in (True, False):
                    prior = self.p25.RefinedQualiaEngineV1_1(enable_episodic_memory=enabled)
                    engine = m.Phase27Engine(mode="phase25_ungated" if enabled else "no_memory",
                                             consolidation_mode="phase25")
                    prior.initialize(fit.p11, fit.p01, fit.marginal)
                    engine.initialize(fit.p11, fit.p01, fit.marginal)
                    for prev, y in zip(stream[:-1], stream[1:]):
                        self.assertEqual(prior.step_update(prev, y), engine.step_update(prev, y))
                        for key in common:
                            self.assertEqual(getattr(prior, key), getattr(engine, key))
                        self.assertEqual([s.__dict__ for s in prior.memory], [s.__dict__ for s in engine.memory])
                    self.assertEqual(prior.recorded_trace.__dict__, engine.recorded_trace.__dict__)

    def test_exact_original_forecasts_and_state_in_legacy_mode(self):
        common = ("p11_fast", "p01_fast", "p11_slow", "p01_slow", "p_marginal_online", "arousal", "valence", "continuity",
                  "stable_steps", "retrievals_accepted", "retrievals_rejected_by_kl_or_rollback", "window")
        for mode in ("phase27_kl_gated", "phase25_ungated", "no_memory"):
            for seed in (123, 20260921, 51000000):
                stream = self.original.generate_scheduled_stream(self.original.mixed_familiar_novel_schedule(), seed)
                old = self.original.Phase27Engine(mode=mode)
                new = m.Phase27Engine(mode=mode, legacy_marginal_order=True, legacy_slow_distance=True)
                new.initialize(.92, .28, 7/9)
                for prev, y in zip(stream[:-1], stream[1:]):
                    self.assertEqual(old.step_update(prev, y), new.step_update(prev, y))
                    for key in common:
                        self.assertEqual(getattr(old, key), getattr(new, key))
                    if mode != "no_memory":
                        self.assertEqual([s.__dict__ for s in old.memory], [s.__dict__ for s in new.memory])

    def test_counter_partition_bounds_and_capacity(self):
        engine = m.Phase27Engine()
        rng = random.Random(123)
        prev = 1
        for t in range(500):
            p11, p01 = (.92, .28) if (t//30) % 2 == 0 else (.15, .85)
            y = int(rng.random() < (p11 if prev else p01))
            p = engine.step_update(prev, y)
            self.assertTrue(0 <= p <= 1)
            self.assertTrue(-1 <= engine.valence <= 1)
            for value in (engine.p11_fast, engine.p01_fast, engine.p11_slow, engine.p01_slow,
                          engine.arousal, engine.continuity, engine.p_marginal_online):
                self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
            self.assertLessEqual(len(engine.window), 6)
            self.assertLessEqual(len(engine.memory), 8)
            prev = y
        rejections = sum(engine.stats[key] for key in ("rejected_fast", "rejected_slow", "rejected_distance",
                         "rejected_missing_rows", "rejected_kl_row1", "rejected_kl_row0"))
        self.assertEqual(rejections+engine.stats["gate_passed_candidates"], engine.stats["candidate_evaluations"])
        self.assertEqual(engine.retrievals_accepted, engine.stats["accepted_retrievals"])
        self.assertEqual(engine.retrievals_rejected_by_kl_or_rollback, engine.stats["raw_retrievals_rejected"])
        tiny = m.Phase27Engine(max_snapshots=1)
        tiny._consolidate(.9, .1, .7)
        self.assertEqual(tiny.stats["capacity_rejections"], 1)
        self.assertEqual(len(tiny.memory), 1)

    def test_gate_inspection_does_not_mutate_stats(self):
        engine = m.Phase27Engine()
        before = copy.deepcopy(engine.__dict__)
        engine.candidate_passes_phase27_gate(m.EpisodicSnapshot(.9, .2, .5), 3, 1, 1, 1, -100., -100.)
        self.assertEqual(engine.__dict__, before)

    def test_invalid_inputs_reject_before_reset_or_update(self):
        for value in (-.1, 1.1, float("nan"), float("inf")):
            engine = m.Phase27Engine()
            before = copy.deepcopy(engine.__dict__)
            with self.assertRaises(ValueError):
                engine.initialize(.9, value, .5)
            self.assertEqual(engine.__dict__, before)
        for options in ({"mode": "wrong"}, {"consolidation_mode": "wrong"}, {"likelihood_mode": "wrong"},
                        {"window_size": 0}, {"window_size": 1.5}, {"max_snapshots": -1},
                        {"kl_row_threshold": -.1}, {"anti_rollback_dist": float("nan")}, {"enable_kl_gate": 1}):
            with self.assertRaises(ValueError):
                m.Phase27Engine(**options)
        for prev, y in ((2, 1), (1, 2), (1, float("nan"))):
            engine = m.Phase27Engine()
            before = copy.deepcopy(engine.__dict__)
            with self.assertRaises(ValueError):
                engine.step_update(prev, y)
            self.assertEqual(engine.__dict__, before)
        for count in (-1, .5, True):
            with self.assertRaises(ValueError):
                m.jeffreys_row_kl(count, 0, .5)
        with self.assertRaises(ValueError):
            m.bernoulli_kl(.5, .5, eps=0.)

    def test_metrics_known_values_and_shapes(self):
        self.assertEqual(m.brier_score([0, 1], [.5, .5]), .25)
        self.assertAlmostEqual(m.binary_cross_entropy([0, 1], [.5, .5]), math.log(2))
        self.assertEqual(m.brier_score([0, 1], [0., 1.]), 0.)
        with self.assertRaises(ValueError):
            m.brier_score([1, 0], [.5])
        with self.assertRaises(ValueError):
            m.binary_cross_entropy([], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
