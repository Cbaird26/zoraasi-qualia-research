#!/usr/bin/env python3
"""Independent mathematics and gate-contract audit of supplied Phase 2.6.

Standard library only. Uses the explicitly syntax-repaired source solely for
code-contract checks; it does not change that source or the original attachment.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "minimal_repair" / "phase2_6_novelty_aware_gate.py"
ENV = None


def counts(path):
    table = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}
    for pair in zip(path, path[1:]):
        table[pair] += 1
    return tuple(table[key] for key in ((1, 1), (1, 0), (0, 1), (0, 0)))


def row_evidence_exact(successes, failures):
    return Fraction(math.factorial(successes) * math.factorial(failures),
                    math.factorial(successes + failures + 1))


def log_evidence(table):
    n11, n10, n01, n00 = table
    return math.log(row_evidence_exact(n11, n10)) + math.log(row_evidence_exact(n01, n00))


def point_log_likelihood(table, p11, p01):
    n11, n10, n01, n00 = table
    return n11 * math.log(p11) + n10 * math.log1p(-p11) + n01 * math.log(p01) + n00 * math.log1p(-p01)


def path_prior_predictive(path):
    """Independent sequential derivation: Beta(1,1) predictive per transition row."""
    rows = [[1, 1], [1, 1]]
    result = Fraction(1)
    for left, right in zip(path, path[1:]):
        result *= Fraction(rows[left][right], sum(rows[left]))
        rows[left][right] += 1
    return result


def expected_brier(q, p):
    return q * (1 - p)**2 + (1 - q) * p**2


def example(table):
    b = point_log_likelihood(table, 0.15, 0.85)
    a = point_log_likelihood(table, 0.92, 0.28)
    novel = log_evidence(table)
    return {"counts_n11_n10_n01_n00": list(table), "stale_B_log_likelihood": b,
            "snapshot_A_log_likelihood": a, "A_minus_B": a - b,
            "uniform_prior_log_evidence": novel, "A_minus_uniform_prior": a - novel}


def candidate_result(table, p11=0.92, p01=0.28, **kwargs):
    eng = ENV["NoveltyAwareEngineV1_2"](**kwargs)
    snap = ENV["EpisodicSnapshot"](p11, p01, 0.5)
    n11, n10, n01, n00 = table
    r1 = ENV["beta_binomial_row_log_evidence"](n11, n10)
    r0 = ENV["beta_binomial_row_log_evidence"](n01, n00)
    fast = point_log_likelihood(table, 0.15, 0.85)
    result, ll = eng._candidate_passes_gate(snap, *table, fast, r1, r0)
    return {"passed": result, "snapshot_log_likelihood": ll, "row1_log_ratio":
            ENV["point_row_log_likelihood"](n11, n10, p11) - r1,
            "row0_log_ratio": ENV["point_row_log_likelihood"](n01, n00, p01) - r0,
            "joint_log_ratio": ll - r1 - r0, "advantage_over_stale_B": ll - fast}


class MathematicalTests(unittest.TestCase):
    def test_row_beta_formula_matches_independent_factorials(self):
        for ones, zeros in itertools.product(range(9), repeat=2):
            log_formula = math.lgamma(ones + 1) + math.lgamma(zeros + 1) - math.lgamma(ones + zeros + 2)
            self.assertAlmostEqual(math.log(row_evidence_exact(ones, zeros)), log_formula, places=12)

    def test_beta_binomial_count_mass_includes_combinations(self):
        for n in range(1, 9):
            pmfs = [math.comb(n, k) * row_evidence_exact(k, n - k) for k in range(n + 1)]
            self.assertEqual(sum(pmfs), 1)
            self.assertTrue(all(p == Fraction(1, n + 1) for p in pmfs))
        self.assertNotEqual(sum(row_evidence_exact(k, 4 - k) for k in range(5)), 1)

    def test_integrated_markov_paths_normalize_conditional_on_start(self):
        for n in range(1, 9):
            for initial in (0, 1):
                total = Fraction(0)
                for tail in itertools.product((0, 1), repeat=n):
                    n11, n10, n01, n00 = counts((initial,) + tail)
                    total += row_evidence_exact(n11, n10) * row_evidence_exact(n01, n00)
                self.assertEqual(total, 1)

    def test_integral_matches_sequential_predictive_for_every_short_path(self):
        for path in itertools.product((0, 1), repeat=7):
            n11, n10, n01, n00 = counts(path)
            integral = row_evidence_exact(n11, n10) * row_evidence_exact(n01, n00)
            self.assertEqual(integral, path_prior_predictive(path))

    def test_transition_flow_identity(self):
        for path in itertools.product((0, 1), repeat=9):
            n11, n10, n01, n00 = counts(path)
            self.assertEqual(n01 - n10, path[-1] - path[0])

    def test_supplied_counts_are_unrealizable(self):
        possible = {counts(path) for path in itertools.product((0, 1), repeat=7)}
        self.assertNotIn((4, 0, 2, 0), possible)
        self.assertIn((3, 1, 2, 0), possible)

    def test_realisable_replacement_preserves_likelihood_example(self):
        path = (0, 1, 0, 1, 1, 1, 1)
        self.assertEqual(counts(path), (3, 1, 2, 0))
        ex = example(counts(path))
        self.assertGreater(ex["A_minus_B"], 0.30)
        self.assertLess(ex["A_minus_uniform_prior"], 0)

    def test_brier_is_noise_plus_squared_bias(self):
        q, p = 0.15, 0.92
        self.assertAlmostEqual((p - q)**2, 0.5929)
        self.assertAlmostEqual(expected_brier(q, p), 0.7204)
        self.assertAlmostEqual(expected_brier(q, p), q * (1 - q) + (p - q)**2)
        self.assertAlmostEqual(expected_brier(q, 0.5), 0.25)

    def test_weighted_retrieval_is_not_full_replacement(self):
        q, before, snap, weight = 0.88, 0.85, 0.28, 0.65
        after = weight * snap + (1 - weight) * before
        self.assertAlmostEqual(after, 0.4795)
        self.assertAlmostEqual((after - q)**2, 0.16040025)
        self.assertAlmostEqual(expected_brier(q, after) - expected_brier(q, before), 0.15950025)
        self.assertNotAlmostEqual((after - q)**2, 0.36)

    def test_blending_penalty_includes_cross_term(self):
        for q, before, snap, weight in itertools.product((0.15, 0.88), (0.3, 0.85), (0.28, 0.92), (0.65, 0.75)):
            after = (1 - weight) * before + weight * snap
            change = 2 * weight * (before - q) * (snap - before) + weight**2 * (snap - before)**2
            self.assertAlmostEqual(expected_brier(q, after) - expected_brier(q, before), change)

    def test_arousal_rate_numerical_claim_is_conditional_arithmetic_only(self):
        self.assertAlmostEqual(0.02 + 0.32 * 0.16**1.4, 0.04459902840793972)
        self.assertLess(0.02 + 0.32 * 0.16**1.4, 0.32)

    def test_model_mixture_evidence_is_not_maximum(self):
        table = (3, 1, 2, 0)
        logls = [point_log_likelihood(table, *p) for p in ((0.92, 0.28), (0.15, 0.85))]
        mixture = math.log(sum(math.exp(ll) for ll in logls) / len(logls))
        self.assertLess(mixture, max(logls))


class ActualGateContractTests(unittest.TestCase):
    def test_repaired_source_evidence_matches_ordered_path_formula(self):
        for ones, zeros in itertools.product(range(9), repeat=2):
            actual = ENV["beta_binomial_row_log_evidence"](ones, zeros)
            self.assertAlmostEqual(actual, math.log(row_evidence_exact(ones, zeros)), places=12)

    def test_single_observed_row_is_skipped_despite_prose(self):
        # Feasible path 0,1,1,1,1,1,1. Row 0 has exactly one unfavorable transition.
        result = candidate_result((5, 0, 1, 0), p01=0.30)
        self.assertLess(result["row0_log_ratio"], -0.25)
        self.assertGreater(result["joint_log_ratio"], 0.15)
        self.assertTrue(result["passed"])

    def test_row_floor_equality_is_accepted_despite_strict_prose(self):
        table = (3, 1, 2, 0)
        n11, n10, n01, n00 = table
        floor = (ENV["point_row_log_likelihood"](n01, n00, 0.28)
                 - ENV["beta_binomial_row_log_evidence"](n01, n00))
        result = candidate_result(table, adv_margin=-100, novelty_bf_threshold=-100, row_bf_floor=floor)
        self.assertEqual(result["row0_log_ratio"], floor)
        self.assertTrue(result["passed"])

    def test_gate_can_accept_feasible_window_generated_by_novel_regime(self):
        result = candidate_result((6, 0, 0, 0))
        self.assertTrue(result["passed"])
        self.assertGreater(0.88**6, 0)  # Novel D can generate six ones after initial one.

    def test_cusum_reset_does_not_cap_scored_brier_at_quarter(self):
        engine = ENV["TunedCUSUMBaseline"](lr=0.14, drift=0, threshold=0.01)
        current = engine.step_update(1, 0)
        self.assertEqual(current, 0.92)
        self.assertGreater((current - 0)**2, 0.25)
        self.assertNotEqual(engine.p11, 0.5)  # Post-reset update happens immediately.


def cooldown_assignments(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if any(isinstance(t, ast.Attribute) and t.attr == "cooldown" for t in node.targets):
                result.append({"line": node.lineno, "value": node.value.value})
    return sorted(result, key=lambda r: r["line"])


def main():
    global ENV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "math_check.json")
    args = parser.parse_args()
    ENV = runpy.run_path(str(args.engine_source))
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(MathematicalTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(ActualGateContractTests),
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    corrected_example = example((3, 1, 2, 0))
    corrected_example["witness_path"] = [0, 1, 0, 1, 1, 1, 1]
    q, p, snap, weight = 0.88, 0.85, 0.28, 0.65
    after = (1 - weight) * p + weight * snap
    report = {
        "status": "executed",
        "tests": {"total": result.testsRun, "mathematical_methods": 12, "actual_gate_contract_methods": 5,
                  "failures": len(result.failures), "errors": len(result.errors)},
        "source_under_contract_test": str(args.engine_source.relative_to(ROOT)) if args.engine_source.is_relative_to(ROOT) else str(args.engine_source),
        "likelihood_interpretation": "Exact ordered Markov-path evidence conditional on start under independent Beta(1,1) row priors; not a raw count PMF.",
        "supplied_unrealisable_example": {**example((4, 0, 2, 0)), "flow_difference_n01_minus_n10": 2,
                                          "admissible_endpoint_differences": [-1, 0, 1]},
        "realisable_replacement_example": corrected_example,
        "weighted_phase25_retrieval": {"q": q, "before": p, "snapshot": snap, "snapshot_weight": weight,
                                       "after": after, "squared_bias_after": (after - q)**2,
                                       "expected_brier_before": expected_brier(q, p),
                                       "expected_brier_after": expected_brier(q, after),
                                       "brier_risk_increase": expected_brier(q, after) - expected_brier(q, p)},
        "stale_prediction": {"q": 0.15, "p": 0.92, "squared_bias": 0.5929, "expected_brier": 0.7204,
                             "realized_brier_y0": 0.8464, "realized_brier_y1": 0.0064},
        "learning_rate_claim": {"rate_at_assumed_arousal_0_16": 0.02 + 0.32 * 0.16**1.4,
                                "actual_phase25_validation_selected_ema_rate": 0.32,
                                "note": "The assumed arousal=.16 plateau floor was not established by the proposal."},
        "singleton_row_counterexample": candidate_result((5, 0, 1, 0), p01=0.30),
        "novel_window_gate_acceptance": {**candidate_result((6, 0, 0, 0)),
                                         "probability_under_novel_D_given_initial_one": 0.88**6},
        "cooldown_constant_assignments": cooldown_assignments(args.engine_source),
        "interpretation": "The numeric gate is a predictive selection heuristic. Its acceptance is not proof of familiarity, novelty rejection is not calibrated OOD detection, and no phenomenological conclusion follows.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"tests": report["tests"], "supplied_example": report["supplied_unrealisable_example"],
                      "valid_example": corrected_example, "weighted_retrieval": report["weighted_phase25_retrieval"],
                      "singleton_row": report["singleton_row_counterexample"],
                      "novel_window_accepted": report["novel_window_gate_acceptance"],
                      "cooldowns": report["cooldown_constant_assignments"]}, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
