"""Deterministic causal/software tests; stochastic superiority is never an invariant."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_5_falsification_suite as m


class FixedIndex:
    def __init__(self, index):
        self.index = index
    def randrange(self, n):
        if not 0 <= self.index < n:
            raise AssertionError("invalid fixture index")
        return self.index


class TestPhase25(unittest.TestCase):
    def setUp(self):
        self.fit = m.generate_training(123)
        self.selected = m.SelectedBaselines(.20, .20, 1.65, .38)

    def test_training_counts_and_disjoint_rng_domains(self):
        self.assertEqual(self.fit.n1+self.fit.n0, 696)
        self.assertEqual(self.fit.p11, (self.fit.n11+1)/(self.fit.n1+2))
        self.assertEqual(self.fit.p01, (self.fit.n01+1)/(self.fit.n0+2))
        self.assertEqual(self.fit.marginal, (self.fit.n11+self.fit.n01+1)/698)
        self.assertEqual(self.fit.training_seed, 1000000123)
        self.assertEqual(self.fit, m.generate_training(123))
        self.assertNotEqual(m.make_rng(123, "schedule").random(), m.make_rng(123, "observations").random())
        m.check_seed_sets(30, 100, 100, 41000000, 51000000, 61000000)
        with self.assertRaises(ValueError):
            m.check_seed_sets(2, 2, 2, 100, 101, 200)
        with self.assertRaises(ValueError):
            m.check_seed_sets(2, 2, 2, 100, 1000000100, 200)

    def test_marginal_advantage_uses_two_pre_observation_predictions(self):
        corrected = m.RefinedQualiaEngineV1_1()
        legacy = m.RefinedQualiaEngineV1_1(legacy_marginal_order=True)
        self.assertEqual(corrected.step_update(1, 1), .5)
        self.assertEqual(corrected.valence, 0.)
        legacy.step_update(1, 1)
        self.assertAlmostEqual(legacy.valence, math.tanh(.6*(-math.log(.525)+math.log(.5))))
        self.assertEqual(corrected.p_marginal_online, .525)

    def test_returned_predictions_precede_outcomes(self):
        for model in m.build_models(self.fit, self.selected, 123).values():
            twin = copy.deepcopy(model)
            expected = model.predict_proba(1)
            self.assertEqual(model.step_update(1, 0), expected)
            self.assertEqual(twin.step_update(1, 1), expected)

    def test_current_and_future_target_donor_outcomes_cannot_change_earlier_predictions(self):
        target = [1, 1, 0, 1, 1, 0, 0, 1, 0]
        donors = {"sync": [1, 0, 0, 1, 0, 1, 1, 1, 0],
                  "offset": [1, 1, 1, 0, 1, 0, 1, 0, 1],
                  "independent": [0, 0, 1, 0, 0, 1, 1, 0, 0]}
        reference = m.run_paths(target, donors, self.fit, self.selected)[0]
        for changed_name in ("target", "sync", "offset", "independent"):
            mutated_target, mutated_donors = copy.deepcopy(target), copy.deepcopy(donors)
            changed = mutated_target if changed_name == "target" else mutated_donors[changed_name]
            for t in range(4, len(changed)):
                changed[t] = 1-changed[t]
            other = m.run_paths(mutated_target, mutated_donors, self.fit, self.selected)[0]
            for arm in reference:
                self.assertEqual(reference[arm][:4], other[arm][:4])

    def test_donor_current_observation_affects_only_following_prediction(self):
        target = [1]*8
        donors = {name: [1]*8 for name in ("sync", "offset", "independent")}
        changed = copy.deepcopy(donors)
        changed["sync"][3] = 0
        first = m.run_paths(target, donors, self.fit, self.selected)[0]
        second = m.run_paths(target, changed, self.fit, self.selected)[0]
        self.assertEqual(first["donor_sync"][:3], second["donor_sync"][:3])
        self.assertNotEqual(first["donor_sync"][3], second["donor_sync"][3])
        self.assertEqual(first["full_engine"], second["full_engine"])

    def test_online_donor_matches_same_time_post_observation_replay(self):
        schedule = m.periodic_schedule(36, 12)
        target = m.generate_stream(schedule, 123)
        donors = {name: m.generate_stream(schedule, 123, f"donor-{name}")
                  for name in ("sync", "offset", "independent")}
        donor = m.RefinedQualiaEngineV1_1(rng_seed=123)
        donor.initialize(self.fit.p11, self.fit.p01, self.fit.marginal)
        trace = []
        for prev, y in zip(donors["sync"][:-1], donors["sync"][1:]):
            donor.step_update(prev, y)
            trace.append(donor.scalars())
        online = m.run_paths(target, donors, self.fit, self.selected, 123)[0]["donor_sync"]
        self.assertEqual(online, m.replay_trace(target, trace, self.fit, 123))
        with self.assertRaises(ValueError):
            m.replay_trace(target, trace[:-1], self.fit)
        with self.assertRaises(ValueError):
            m.run_paths(target, {**donors, "sync": donors["sync"][:-1]}, self.fit, self.selected)

    def test_reinitialization_resets_states_memory_trace_and_rng(self):
        old, fresh = m.RefinedQualiaEngineV1_1(rng_seed=5), m.RefinedQualiaEngineV1_1(rng_seed=5)
        for i in range(20):
            old.step_update(i % 2, (i+1) % 2)
        old.initialize(.8, .2, .6)
        fresh.initialize(.8, .2, .6)
        for field in ("p11_fast", "p01_fast", "arousal", "valence", "continuity", "memory", "window", "recorded_trace", "stats"):
            self.assertEqual(getattr(old, field), getattr(fresh, field))
        self.assertEqual(old.rng.random(), fresh.rng.random())

    def test_memory_instance_isolation_and_zero_capacity(self):
        one, two = m.RefinedQualiaEngineV1_1(), m.RefinedQualiaEngineV1_1()
        one.memory[0].p11 = .9
        one.window.append((1, 0))
        self.assertEqual(two.memory[0].p11, .5)
        self.assertEqual(two.window, [])
        empty = m.RefinedQualiaEngineV1_1(max_snapshots=0)
        disabled = m.RefinedQualiaEngineV1_1(enable_episodic_memory=False)
        self.assertEqual(empty.memory, [])
        self.assertEqual(disabled.memory, [])
        for i in range(100):
            empty.step_update(1, 1)
        self.assertEqual(empty.memory, [])

    def test_no_update_preserves_extreme_probabilities(self):
        engine = m.RefinedQualiaEngineV1_1(False, False, fixed_lr=0.)
        engine.initialize(.999, .001, .5)
        baseline = m.PlainEMA(.999, .001, 0.)
        for prev, y in ((1, 0), (0, 1), (1, 1), (0, 0)):
            expected = .999 if prev else .001
            self.assertEqual(engine.step_update(prev, y), expected)
            self.assertEqual(baseline.step_update(prev, y), expected)
        self.assertEqual((engine.p11_fast, engine.p01_fast), (.999, .001))

    def test_random_snapshot_control_chooses_random_candidate_and_keeps_gate(self):
        models = []
        for mode in ("best", "random"):
            model = m.RefinedQualiaEngineV1_1(enable_affective_plasticity=False,
                                            fixed_lr=0., retrieval_mode=mode)
            model.memory = [m.EpisodicSnapshot(.95, .2, .5), m.EpisodicSnapshot(.05, .2, .5)]
            model.window = [(1, 1)]*5
            model.rng = FixedIndex(1)
            model.step_update(1, 1, (.8, 0., .2))
            models.append(model)
        best, random_control = models
        self.assertEqual(best.stats["retrieval_eligible"], 1)
        self.assertEqual(random_control.stats["retrieval_eligible"], 1)
        self.assertEqual(best.stats["candidate_draws"], 1)
        self.assertEqual(random_control.stats["candidate_draws"], 1)
        self.assertEqual(best.last_candidate_index, 0)
        self.assertEqual(random_control.last_candidate_index, 1)
        self.assertEqual(best.last_best_index, random_control.last_best_index)
        self.assertEqual(best.stats["retrievals"], 1)
        self.assertEqual(random_control.stats["retrievals"], 0)  # same likelihood gate rejects worse random candidate
        self.assertEqual(random_control.p11_fast, .5)

    def test_capacity_and_consolidation_event_counters(self):
        engine = m.RefinedQualiaEngineV1_1(max_snapshots=2)
        engine._consolidate(.95, .1, .7)
        engine._consolidate(.94, .11, .8)
        engine._consolidate(.1, .9, .6)
        self.assertEqual(len(engine.memory), 2)
        self.assertEqual(engine.stats["snapshot_stores"], 1)
        self.assertEqual(engine.stats["snapshot_merges"], 1)
        self.assertEqual(engine.stats["capacity_rejections"], 1)

    def test_bounded_states_through_switches(self):
        engine = m.RefinedQualiaEngineV1_1()
        stream = m.generate_stream(m.periodic_schedule(450, 15), 123)
        for prev, y in zip(stream[:-1], stream[1:]):
            p = engine.step_update(prev, y)
            self.assertTrue(0 <= p <= 1)
            m.validate_scalars(engine.scalars())
            self.assertLessEqual(len(engine.window), 6)
            self.assertLessEqual(len(engine.memory), 8)
        self.assertEqual(len(engine.recorded_trace.arousal), 450)

    def test_fixed_offset_and_independent_dwell_schedules(self):
        shifted = m.periodic_schedule(360, 45, 23)
        self.assertEqual([(b.dwell, b.label) for b in shifted],
                         [(22, "A"), (45, "B"), (45, "A"), (45, "B"), (45, "A"),
                          (45, "B"), (45, "A"), (45, "B"), (23, "A")])
        self.assertEqual(m.rotate_schedule(m.periodic_schedule(360)), shifted)
        first = m.stochastic_schedule(123, "recipient-schedule")
        second = m.stochastic_schedule(123, "independent-donor-schedule")
        self.assertEqual(sum(b.dwell for b in first), 450)
        self.assertEqual(sum(b.dwell for b in second), 450)
        self.assertNotEqual(first, second)
        self.assertEqual(first, m.stochastic_schedule(123, "recipient-schedule"))
        self.assertTrue(all(25 <= b.dwell <= 65 for b in first[:-1]))

    def test_initial_state_distribution_controls_and_scope_slices(self):
        schedule = m.periodic_schedule(450)
        self.assertEqual(m.generate_stream(schedule, 3, q0=0.)[0], 0)
        self.assertEqual(m.generate_stream(schedule, 3, q0=1.)[0], 1)
        schedule, _ = m.make_scenario("familiar_novel", 3)
        scopes, blocks = m.indexes_and_blocks(schedule)
        self.assertEqual(scopes["novel_C_D"], list(range(180, 270)))
        self.assertEqual(scopes["postnovel_revisited"], list(range(270, 360)))
        self.assertEqual(scopes["initial_30"], list(range(30)))
        self.assertEqual(blocks[-1]["occurrence_in_test"], 3)

    def test_legacy_mode_exactly_reproduces_original_equations(self):
        path = Path(__file__).resolve().parents[1] / "original" / "phase2_5_falsification_suite.py"
        spec = importlib.util.spec_from_file_location("original_phase25", path)
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        schedule = [(45, *m.A), (45, *m.B)]*4
        stream = original.generate_scheduled_stream(schedule, 20260921)
        before = original.RefinedQualiaEngineV1_1()
        after = m.RefinedQualiaEngineV1_1(legacy_marginal_order=True)
        after.initialize(.92, .28, 7/9)
        for prev, y in zip(stream[:-1], stream[1:]):
            self.assertEqual(before.step_update(prev, y), after.step_update(prev, y))
            for key in ("p11_fast", "p01_fast", "p_marginal_online", "arousal", "valence", "continuity"):
                self.assertEqual(getattr(before, key), getattr(after, key))

    def test_validation_grid_endpoints_and_test_independence(self):
        with patch.object(m, "evaluate_seed", side_effect=AssertionError("test data accessed during validation")):
            manifest = m.tune_baselines(2, 41000000)
        rates = [r["params"]["rate"] for r in manifest["candidate_results"]["ema"]]
        self.assertEqual(rates, [i/100 for i in range(1, 51)])
        self.assertEqual(manifest["validation_seeds"], [41000000, 41000001])
        expected = min(manifest["candidate_results"]["ema"], key=lambda r: r["mean_brier"])["params"]["rate"]
        self.assertEqual(manifest["selected"]["ema_rate"], expected)
        frozen = m.SelectedBaselines(**manifest["selected"])
        with self.assertRaises(Exception):
            frozen.ema_rate = .99

    def test_metrics_and_paired_interval_arithmetic(self):
        self.assertEqual(m.score([0, 1], [.5, .5])["brier"], .25)
        self.assertAlmostEqual(m.score([0, 1], [.5, .5])["bce"], math.log(2))
        self.assertEqual(m.score([0, 1], [0, 1])["brier"], 0.)
        stats = m.mean_ci([1., 2., 3.])
        self.assertEqual(stats["mean"], 2.)
        self.assertAlmostEqual(stats["normal_mc_ci95_halfwidth"], 1.96/math.sqrt(3))
        self.assertEqual(m.mean_ci([.2-.3]*3)["normal_mc_ci95_halfwidth"], 0.)

    def test_invalid_input_validation(self):
        for p in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                m.RefinedQualiaEngineV1_1(fixed_lr=p)
            with self.assertRaises(ValueError):
                m.RefinedQualiaEngineV1_1().initialize(.9, p, .5)
        with self.assertRaises(ValueError):
            m.RefinedQualiaEngineV1_1(max_snapshots=-1)
        with self.assertRaises(ValueError):
            m.RefinedQualiaEngineV1_1(retrieval_mode="unrecognized")
        with self.assertRaises(ValueError):
            m.RefinedQualiaEngineV1_1().step_update(1, 2)
        with self.assertRaises(ValueError):
            m.RefinedQualiaEngineV1_1().step_update(1, 1, (.2, .4))
        with self.assertRaises(ValueError):
            m.RefinedQualiaEngineV1_1().step_update(1, 1, (.2, 2., .4))
        with self.assertRaises(ValueError):
            m.score([1, 0], [.5])
        with self.assertRaises(ValueError):
            m.Block(0, .9, .1, "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
