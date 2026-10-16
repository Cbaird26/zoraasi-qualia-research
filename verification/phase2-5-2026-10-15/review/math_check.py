#!/usr/bin/env python3
"""Independent Phase 2.5 claim audit, standard library only.

Run from any directory. Optional --phase24-source checks the previously audited
engine's unused stored continuity. No source file is changed or imported as main.
The actual Phase 2.5 source beside this bundle is used for two chronology checks.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import runpy
import unittest

A, B, LAMBDA, PI, H = 0.28, 0.92, 0.64, 7.0 / 9.0, 29
METRICS = ("brier", "bce_nats")
ARMS = ("stateful", "memoryless", "reset", "deranged")


def risk(q, p):
    return {"brier": q * (1 - p) ** 2 + (1 - q) * p**2,
            "bce_nats": -q * math.log(p) - (1 - q) * math.log1p(-p)}


def mix(q, zero, one):
    return {k: (1 - q) * zero[k] + q * one[k] for k in METRICS}


def marginal(q0, t):
    return PI + (q0 - PI) * LAMBDA**t


def per_step(q0, t, qbar):
    previous, target = marginal(q0, t - 1), marginal(q0, t)
    return {
        "stateful": mix(previous, risk(A, A), risk(B, B)),
        "memoryless": risk(target, qbar),
        "reset": mix(q0, risk(PI * (1 - LAMBDA**t), A), risk(PI + (1 - PI) * LAMBDA**t, B)),
        "deranged": mix(previous, risk(target, A), risk(target, B)),
    }


def expected(q0, horizon=H):
    qbar = math.fsum(marginal(q0, t) for t in range(1, horizon + 1)) / horizon
    steps = [per_step(q0, t, qbar) for t in range(1, horizon + 1)]
    average = {arm: {k: math.fsum(s[arm][k] for s in steps) / horizon for k in METRICS} for arm in ARMS}
    return {"q0": q0, "horizon": horizon, "pooled_target_marginal": qbar,
            "average": average, "per_step": [{"t": t + 1, **s} for t, s in enumerate(steps)]}


def asymptotic(q0):
    """t -> infinity pointwise; distinct from averaging a fixed 29 steps."""
    return {"stateful": mix(PI, risk(A, A), risk(B, B)),
            "memoryless_fixed_H29_prior": risk(PI, expected(q0)["pooled_target_marginal"]),
            "memoryless_stationary_prior": risk(PI, PI),
            "reset": mix(q0, risk(PI, A), risk(PI, B)),
            "deranged": mix(PI, risk(PI, A), risk(PI, B))}


def crossover(metric):
    """Delta(q0)=mean reset risk - mean deranged risk is quadratic in q0."""
    def delta(q):
        s = expected(q)["average"]
        return s["reset"][metric] - s["deranged"][metric]
    c, dh, d1 = delta(0), delta(0.5), delta(1)
    a = 2 * (d1 + c - 2 * dh)
    b = d1 - c - a
    discriminant = b * b - 4 * a * c
    roots = sorted(((-b - math.sqrt(discriminant)) / (2 * a),
                    (-b + math.sqrt(discriminant)) / (2 * a)))
    return {"quadratic_a": a, "quadratic_b": b, "quadratic_c": c,
            "all_roots": roots, "root_in_unit_interval": next(r for r in roots if 0 <= r <= 1)}


def enumerate_risks(q0, horizon=4):
    """Independent full own-path and donor-path enumeration, no marginal formula."""
    paths = []
    for seq in itertools.product((0, 1), repeat=horizon + 1):
        mass = q0 if seq[0] else 1 - q0
        for prev, cur in zip(seq, seq[1:]):
            p = B if prev else A
            mass *= p if cur else 1 - p
        paths.append((seq, mass))
    qbar = math.fsum(mass * sum(seq[1:]) for seq, mass in paths) / horizon
    result = {a: {k: 0.0 for k in METRICS} for a in ARMS}
    for seq, mass in paths:
        for t in range(1, horizon + 1):
            forecasts = {"stateful": B if seq[t - 1] else A,
                         "memoryless": qbar, "reset": B if seq[0] else A}
            for arm, p in forecasts.items():
                realized = risk(seq[t], p)
                for k in METRICS:
                    result[arm][k] += mass * realized[k] / horizon
        for donor, donor_mass in paths:
            for t in range(1, horizon + 1):
                realized = risk(seq[t], B if donor[t - 1] else A)
                for k in METRICS:
                    result["deranged"][k] += mass * donor_mass * realized[k] / horizon
    return result


def blend_excess(q, f, m, w):
    alpha = 1 - w
    return 2 * alpha * (f - q) * (m - f) + alpha**2 * (m - f)**2


class MathTests(unittest.TestCase):
    def test_exhaustive_independent_paths_all_four_initializations(self):
        for q0 in (0, 0.5, PI, 1):
            direct, formula = enumerate_risks(q0), expected(q0, 4)["average"]
            for arm in ARMS:
                for metric in METRICS:
                    self.assertAlmostEqual(direct[arm][metric], formula[arm][metric], places=12)

    def test_marginals_match_transition_recursion(self):
        for q0 in (0, 0.5, PI, 1):
            q = q0
            for t in range(H + 1):
                self.assertAlmostEqual(q, marginal(q0, t), places=14)
                q = (1 - q) * A + q * B

    def test_stationary_reset_closed_forms(self):
        v, g = PI * (1 - PI), (1 - LAMBDA**H) / (H * (1 - LAMBDA))
        logit_difference = math.log(B / (1 - B)) - math.log(A / (1 - A))
        deranged_bce = mix(PI, risk(PI, A), risk(PI, B))["bce_nats"]
        row = expected(PI)["average"]["reset"]
        self.assertAlmostEqual(row["brier"], v * (1 + LAMBDA**2) - 2 * v * LAMBDA**2 * g, places=14)
        self.assertAlmostEqual(row["bce_nats"], deranged_bce - v * LAMBDA * logit_difference * g, places=14)

    def test_pointwise_asymptotes(self):
        for q0 in (0, 0.5, PI, 1):
            direct = per_step(q0, 100, expected(q0)["pooled_target_marginal"])
            limit = asymptotic(q0)
            for arm in ("stateful", "reset", "deranged"):
                for metric in METRICS:
                    self.assertAlmostEqual(direct[arm][metric], limit[arm][metric], places=13)

    def test_quadratic_difference_reconstruction(self):
        for metric in METRICS:
            c = crossover(metric)
            for q0 in (0.1, 0.3, 0.68, PI, 0.95):
                row = expected(q0)["average"]
                poly = c["quadratic_a"] * q0**2 + c["quadratic_b"] * q0 + c["quadratic_c"]
                self.assertAlmostEqual(poly, row["reset"][metric] - row["deranged"][metric], places=13)

    def test_crossover_zero_and_side_signs(self):
        for metric in METRICS:
            root = crossover(metric)["root_in_unit_interval"]
            def delta(q):
                row = expected(q)["average"]
                return row["reset"][metric] - row["deranged"][metric]
            self.assertAlmostEqual(delta(root), 0, places=13)
            self.assertGreater(delta(root - 0.01), 0)
            self.assertLess(delta(root + 0.01), 0)

    def test_metric_rankings_differ_between_crossovers(self):
        row = expected(0.68)["average"]
        self.assertGreater(row["reset"]["brier"], row["deranged"]["brier"])
        self.assertLess(row["reset"]["bce_nats"], row["deranged"]["bce_nats"])

    def test_brier_noise_plus_bias_identity(self):
        for q, p in itertools.product((0.1, 0.5, 0.82), (0.2, 0.7, 0.9)):
            self.assertAlmostEqual(risk(q, p)["brier"], q * (1 - q) + (p - q)**2, places=14)

    def test_blend_excess_identity(self):
        for q, f, m, w in itertools.product((0.18, 0.82), (0.1, 0.7), (0.5, PI), (0.15, 0.5, 1)):
            direct = risk(q, w * f + (1 - w) * m)["brier"] - risk(q, f)["brier"]
            self.assertAlmostEqual(direct, blend_excess(q, f, m, w), places=14)

    def test_neutral_shrinkage_is_not_guaranteed_better(self):
        q, f, w = 0.82, 0.82, 0.5
        neutral = risk(q, w * f + (1 - w) * 0.5)["brier"]
        stale = risk(q, w * f + (1 - w) * PI)["brier"]
        self.assertGreater(neutral, stale)
        self.assertGreater(stale, risk(q, f)["brier"])

    def test_current_label_update_optimistically_scores_online_marginal(self):
        for y in (0, 1):
            old, eta = PI, 0.05
            updated = (1 - eta) * old + eta * y
            self.assertLess(risk(y, updated)["bce_nats"], risk(y, old)["bce_nats"])
            self.assertAlmostEqual(risk(y, updated)["brier"], (1 - eta)**2 * risk(y, old)["brier"], places=14)


def check_phase25_chronology(source):
    env = runpy.run_path(str(source))
    cls = env["RefinedQualiaEngineV1_1"]
    low, high = cls(), cls()
    low_now = low.step_update(1, 1, (0.0, 0.0, 1.0))
    high_now = high.step_update(1, 1, (1.0, 0.0, 0.0))
    assert low_now == high_now == 0.92
    assert low.predict_proba(1) != high.predict_proba(1)
    donor_sequences = ([1, 0, 1], [1, 1, 1])
    traces = []
    for seq in donor_sequences:
        donor = cls()
        for prev, cur in zip(seq, seq[1:]):
            donor.step_update(prev, cur)
        traces.append(donor.recorded_trace)
    recipients = []
    for trace in traces:
        recipient = cls()
        predictions = []
        for t in range(2):
            scalars = (trace.arousal[t], trace.valence[t], trace.continuity[t])
            predictions.append(recipient.step_update(1, 1, scalars))
        recipients.append(predictions)
    assert recipients[0][0] == recipients[1][0]
    assert recipients[0][1] != recipients[1][1]
    return {"status": "passed", "checks": 2,
            "extreme_scalar_transfer": {"current_prediction": low_now,
                                        "next_prediction_low_arousal": low.predict_proba(1),
                                        "next_prediction_high_arousal": high.predict_proba(1)},
            "donor_outcome_t_perturbation_recipient_predictions": recipients,
            "interpretation": "Donor outcome t may change recipient p_(t+1), but does not change scored p_t."}


def check_phase24_continuity(source):
    env = runpy.run_path(str(source))
    cls = env["SyntheticQualiaEngineV1"]
    paths = 0
    transitions = 0
    for seq in itertools.product((0, 1), repeat=8):
        first = cls()
        first.initialize_from_phase23(B, A, PI)
        second = copy.deepcopy(first)
        for prev, cur in zip(seq, seq[1:]):
            first.state.continuity = 0.0
            second.state.continuity = 1.0
            assert first.step_update(prev, cur) == second.step_update(prev, cur)
            assert asdict(first) == asdict(second)
            transitions += 1
        paths += 1
    return {"status": "passed", "checks": 1, "paths": paths, "paired_transitions": transitions,
            "interpretation": "Changing only stored continuity never changes the next prediction or updated state; old continuity is overwritten before use."}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "results" / "math_check.json")
    parser.add_argument("--phase25-source", type=Path, default=root / "original" / "phase2_5_falsification_suite.py")
    parser.add_argument("--phase24-source", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MathTests))
    if not result.wasSuccessful():
        return 1
    original_table = {
        "1.0": (1.0, 0.1818, 0.2372),
        "7/9": (PI, 0.2301, 0.2436),
        "0.5": (0.5, 0.2912, 0.2504),
        "0.0": (0.0, 0.4005, 0.2627),
    }
    table = {}
    for label, (q0, pasted_reset, pasted_deranged) in original_table.items():
        ex = expected(q0)
        table[label] = {**ex, "pointwise_t_infinity": asymptotic(q0),
                        "pasted_brier": {"reset": pasted_reset, "deranged": pasted_deranged},
                        "pasted_minus_exact": {"reset": pasted_reset - ex["average"]["reset"]["brier"],
                                               "deranged": pasted_deranged - ex["average"]["deranged"]["brier"]}}
    report = {
        "status": "executed", "mathematical_tests": {"count": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)},
        "scope": "Fixed known p11=.92, p01=.28; 29 equally weighted transitions; no finite-training noise or consciousness inference.",
        "initialization_table": table,
        "crossover": {m: crossover(m) for m in METRICS},
        "conditional_stationary_limits": {"initial_state_zero": risk(PI, A), "initial_state_one": risk(PI, B)},
        "blend_brier_excess_formula": "2(1-w)(f-q)(m-f)+(1-w)^2(m-f)^2",
        "neutral_prior_counterexample": {"q": 0.82, "f": 0.82, "w": 0.5,
                                         "unblended": risk(0.82, 0.82),
                                         "stationary_prior_blend": risk(0.82, 0.5 * 0.82 + 0.5 * PI),
                                         "neutral_prior_blend": risk(0.82, 0.5 * 0.82 + 0.5 * 0.5)},
        "phase25_chronology": check_phase25_chronology(args.phase25_source),
        "phase24_stored_continuity": check_phase24_continuity(args.phase24_source) if args.phase24_source else {"status": "not_requested"},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for label, row in table.items():
        r, d = row["average"]["reset"], row["average"]["deranged"]
        print(f"q0={label:>3}: reset Brier={r['brier']:.12f}, BCE={r['bce_nats']:.12f}; deranged Brier={d['brier']:.12f}, BCE={d['bce_nats']:.12f}")
    for metric in METRICS:
        print(f"{metric} finite-H crossover q0={report['crossover'][metric]['root_in_unit_interval']:.12f}")
    print("Phase 2.5 chronology checks: 2 passed")
    if args.phase24_source:
        print("Phase 2.4 stored-continuity check: passed, 256 paths / 1792 paired transitions")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
