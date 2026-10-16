"""Deterministic correctness and causal-boundary tests, not superiority assertions."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_4_functional_state_engine as m


class TestPhase24(unittest.TestCase):
    def setUp(self):
        self.fit = m.fit_laplace([[1, 1, 0, 0, 1], [0, 1, 1, 1, 0]])

    def test_laplace_fit_exact_counts(self):
        self.assertEqual((self.fit.n11, self.fit.n1, self.fit.n01, self.fit.n0), (3, 5, 2, 3))
        self.assertEqual((self.fit.p11, self.fit.p01, self.fit.marginal), (4 / 7, 3 / 5, 6 / 10))

    def test_independent_training_has_requested_count(self):
        fit = m.generate_training(55)
        self.assertEqual(fit.n1 + fit.n0, 696)
        self.assertEqual(fit, m.generate_training(55))
        self.assertNotEqual(m.make_rng(55, "independent-training").random(),
                            m.make_rng(55, "test-stream-0").random())

    def test_update_returns_pre_outcome_prediction(self):
        for model in m.build_models(self.fit).values():
            twin = copy.deepcopy(model)
            expected = model.predict_proba(1)
            self.assertEqual(model.step_update(1, 0), expected)
            self.assertEqual(twin.step_update(1, 1), expected)

    def test_current_and_future_outcomes_do_not_change_earlier_predictions(self):
        streams = [[1, 1, 0, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0]]
        changed = copy.deepcopy(streams)
        for stream in changed:
            for t in range(4, len(stream)):
                stream[t] = 1 - stream[t]
        first, second = m.prediction_paths(streams, self.fit), m.prediction_paths(changed, self.fit)
        for arm in first:
            for i in range(2):
                # Index 3 predicts outcome at t=4; that outcome has not yet been seen.
                self.assertEqual(first[arm][i][:4], second[arm][i][:4])

    def test_reinitialization_matches_fresh_instance(self):
        engine = m.SyntheticQualiaEngineV1()
        engine.initialize_from_estimates(.92, .28, 7 / 9)
        for t in range(80):
            engine.step_update(t % 2, (t + 1) % 2)
        engine.initialize_from_estimates(.77, .17, .58)
        fresh = m.SyntheticQualiaEngineV1()
        fresh.initialize_from_estimates(.77, .17, .58)
        self.assertEqual(engine, fresh)
        self.assertEqual(engine.recent_transitions, [])

    def test_mutable_memory_is_isolated(self):
        first, second = m.SyntheticQualiaEngineV1(), m.SyntheticQualiaEngineV1()
        first.initialize_from_estimates(.9, .2, .6)
        second.initialize_from_estimates(.9, .2, .6)
        first.episodic_memory[0].p11 = .3
        first.recent_transitions.append((1, 0))
        first.state.arousal = .9
        self.assertEqual(second.episodic_memory[0].p11, .9)
        self.assertEqual(second.recent_transitions, [])
        self.assertEqual(second.state.arousal, .25)

    def test_state_and_memory_bounds(self):
        for slots in (0, 1, 8):
            engine = m.SyntheticQualiaEngineV1(max_memory_slots=slots)
            engine.initialize_from_estimates(.92, .28, 7 / 9)
            stream = m.generate_stream(m.regime_schedule("periodic_15", 500), m.make_rng(4, "bounds"))
            for prev, current in zip(stream[:-1], stream[1:]):
                pred = engine.step_update(prev, current)
                self.assertTrue(0 <= pred <= 1)
                self.assertTrue(-1 <= engine.state.valence <= 1)
                for value in (engine.state.arousal, engine.state.self_confidence, engine.state.continuity,
                              engine.p11_fast, engine.p01_fast, engine.p11_slow, engine.p01_slow):
                    self.assertTrue(math.isfinite(value) and 0 <= value <= 1)
                self.assertLessEqual(len(engine.episodic_memory), slots)
                self.assertLessEqual(len(engine.recent_transitions), 6)

    def test_zero_rate_never_clips_frozen_parameters(self):
        engine = m.SyntheticQualiaEngineV1(False, False, False, fixed_learning_rate=0)
        engine.initialize_from_estimates(.999, .001, .5)
        baseline = m.PlainMarkovEMA(.999, .001, 0)
        for prev, current in ((1, 0), (0, 1), (1, 1), (0, 0)):
            expected = .999 if prev else .001
            self.assertEqual(engine.step_update(prev, current), expected)
            self.assertEqual(baseline.step_update(prev, current), expected)
        self.assertEqual((engine.p11_fast, engine.p01_fast), (.999, .001))
        self.assertEqual((baseline.p11, baseline.p01), (.999, .001))

    def test_plain_baselines_use_only_declared_information(self):
        model = m.PlainMarkovEMA(.8, .2, .08)
        self.assertEqual(model.step_update(1, 0), .8)
        self.assertAlmostEqual(model.p11, .736)
        self.assertEqual(model.p01, .2)
        marginal = m.MemorylessEMA(.7, 0)
        self.assertEqual(marginal.predict_proba(0), marginal.predict_proba(1))
        for y in (0, 1, 1):
            self.assertEqual(marginal.step_update(1, y), .7)

    def test_sham_fixed_derangement_no_self_donors(self):
        for count in (2, 3, 4, 9):
            sigma = m.fixed_derangement(count)
            self.assertEqual(sorted(sigma), list(range(count)))
            self.assertTrue(all(i != j for i, j in enumerate(sigma)))

    def test_sham_copies_only_four_fields_without_mutating_donors(self):
        groups = [m.build_models(self.fit) for _ in range(3)]
        for i, group in enumerate(groups):
            group["full_engine"].state = m.InternalStateVector(.1 * i, .2 * i, .3 * i, .4 * i)
            group["deranged_state_vector"].step_update(1, i % 2)
        own_before = [copy.deepcopy(g["deranged_state_vector"]) for g in groups]
        donor_before = [copy.deepcopy(g["full_engine"]) for g in groups]
        sigma = m.inject_deranged_states(groups)
        for i, donor in enumerate(sigma):
            recipient = groups[i]["deranged_state_vector"]
            self.assertEqual(recipient.state, donor_before[donor].state)
            self.assertIsNot(recipient.state, groups[donor]["full_engine"].state)
            self.assertEqual(recipient.recent_transitions, own_before[i].recent_transitions)
            self.assertEqual(recipient.episodic_memory, own_before[i].episodic_memory)
            self.assertEqual(recipient.p11_fast, own_before[i].p11_fast)
            self.assertEqual(recipient.p01_slow, own_before[i].p01_slow)
            self.assertEqual(groups[donor]["full_engine"], donor_before[donor])
        groups[0]["deranged_state_vector"].state.arousal = .999
        self.assertEqual(groups[1]["full_engine"], donor_before[1])

    def test_original_normal_path_dynamics_preserved(self):
        path = Path(__file__).resolve().parents[1] / "original" / "synthetic_qualia_engine_v1.py"
        spec = importlib.util.spec_from_file_location("uploaded_original_phase24", path)
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        stream = original.generate_regime_switching_stream()
        for options in ({}, {"enable_episodic_memory": False}, {"enable_affective_plasticity": False},
                        {"enable_self_model": False}, {"enable_affective_plasticity": False,
                         "enable_episodic_memory": False, "enable_self_model": False, "fixed_learning_rate": 0}):
            before, after = original.SyntheticQualiaEngineV1(**options), m.SyntheticQualiaEngineV1(**options)
            before.initialize_from_phase23(.92, .28, 7 / 9)
            after.initialize_from_phase23(.92, .28, 7 / 9)
            for prev, current in zip(stream[:-1], stream[1:]):
                self.assertEqual(before.step_update(prev, current), after.step_update(prev, current))
                self.assertEqual(before.state.__dict__, after.state.__dict__)

    def test_invalid_inputs_fail_before_mutation(self):
        for value in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                m.SyntheticQualiaEngineV1(fixed_learning_rate=value)
            engine = m.SyntheticQualiaEngineV1()
            before = copy.deepcopy(engine)
            with self.assertRaises(ValueError):
                engine.initialize_from_estimates(.9, value, .5)
            self.assertEqual(before, engine)
        for value in (2, -1, None, float("nan")):
            with self.assertRaises(ValueError):
                m.SyntheticQualiaEngineV1().step_update(1, value)
        for eps in (0, .5, 1, float("nan")):
            with self.assertRaises(ValueError):
                m.SyntheticQualiaEngineV1().step_update(1, 0, eps)
        with self.assertRaises(ValueError):
            m.SyntheticQualiaEngineV1(max_memory_slots=-1)
        with self.assertRaises(ValueError):
            m.score([1, 0], [.5])
        with self.assertRaises(ValueError):
            m.prediction_paths([[1, 0], [1, 1, 0]], self.fit)
        with self.assertRaises(ValueError):
            m.fixed_derangement(1)

    def test_metrics_at_known_values(self):
        self.assertEqual(m.score([0, 1], [0, 1])["brier"], 0)
        self.assertEqual(m.score([0, 1], [.5, .5])["brier"], .25)
        self.assertAlmostEqual(m.score([0, 1], [.5, .5])["bce"], math.log(2))

    def test_seed_reproducibility_and_arm_presence(self):
        first = m.evaluate_seed(1234, "periodic_15", horizon=40)
        self.assertEqual(first, m.evaluate_seed(1234, "periodic_15", horizon=40))
        self.assertEqual(len(first["metrics"]), 11)
        self.assertIn("plain_fixed_ema", first["metrics"])
        self.assertIn("deranged_state_vector", first["metrics"])

    def test_monte_carlo_intervals_are_seed_level_and_paired(self):
        values = m.mean_interval([1, 2, 3])
        self.assertEqual(values["n_seeds"], 3)
        self.assertEqual(values["mean"], 2)
        self.assertAlmostEqual(values["normal_mc_ci95_halfwidth"], 1.96 / math.sqrt(3))
        records = []
        for seed, offset in enumerate((10, 20, 30)):
            records.append({"split": "primary", "scenario": "stationary_a", "seed": seed,
                            "metrics": {"full_engine": {"brier": offset, "bce": 2 * offset},
                                        "plain_fixed_ema": {"brier": offset + 1, "bce": 2 * offset + 2}}})
        paired = m.summarize(records)["primary"]["stationary_a"]["paired_full_minus_comparator"]["plain_fixed_ema"]
        self.assertEqual(paired["brier"]["mean"], -1)
        self.assertEqual(paired["brier"]["normal_mc_ci95_halfwidth"], 0)
        self.assertEqual(paired["bce"]["mean"], -2)

    def test_holdout_seeds_must_be_disjoint(self):
        with self.assertRaises(ValueError):
            m.run_benchmark(2, 2, 100, 101)


if __name__ == "__main__":
    unittest.main(verbosity=2)
