#!/usr/bin/env python3
"""Independent exact expected-loss verification for the Phase 2.3 toy process.

Standard library only. No imported experiment implementation. Runs exhaustive
short-path checks against closed-form expectations, then writes the full H=29
known-parameter references. A known transition probability is used even by the
deliberately misspecified reset/deranged predictors; "oracle" here is not a claim
that those predictors are optimal for the information they receive.

Run: python3 review/math_verify.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import sys
import unittest

ARMS = ("stateful_markov", "memoryless_marginal", "reset_each_trial", "deranged_memory")
METRICS = ("brier", "bce_nats")


@dataclass(frozen=True)
class Process:
    p11: float = 0.92
    p01: float = 0.28
    initial_one_probability: float = 0.5
    horizon: int = 29

    def __post_init__(self):
        if not (0 < self.p11 < 1 and 0 < self.p01 < 1):
            raise ValueError("Interior transition probabilities required for finite logs.")
        if not 0 <= self.initial_one_probability <= 1:
            raise ValueError("Invalid initial-state probability.")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("Positive integer prediction horizon required.")

    @property
    def stationary(self):
        return self.p01 / (1.0 - self.p11 + self.p01)

    @property
    def eigenvalue(self):
        return self.p11 - self.p01

    def marginal(self, t):
        return self.stationary + (self.initial_one_probability - self.stationary) * self.eigenvalue**t

    def probability(self, state):
        return self.p11 if state else self.p01


def expected_loss(outcome_probability, prediction):
    """Exact risk for Y~Bernoulli(outcome_probability), constant prediction."""
    q, p = outcome_probability, prediction
    return {
        "brier": q * (1 - p) ** 2 + (1 - q) * p**2,
        "bce_nats": -q * math.log(p) - (1 - q) * math.log1p(-p),
    }


def mixture(weight, loss_zero, loss_one):
    return {m: (1 - weight) * loss_zero[m] + weight * loss_one[m] for m in METRICS}


def reference(process, marginal_prediction=None):
    """Expected risks for the specified four predictors, with true p01 and p11.

    The fitted pooled marginal converges to the mean of target marginals at the
    fixed trial horizon as the number of independent training trials increases.
    Supplying marginal_prediction=pi instead is a separate stationary-prior
    comparison, not the nonstationary training limit.
    """
    cfg = process
    pi, lam, mu0 = cfg.stationary, cfg.eigenvalue, cfg.initial_one_probability
    pooled = math.fsum(cfg.marginal(t) for t in range(1, cfg.horizon + 1)) / cfg.horizon
    marginal_prediction = pooled if marginal_prediction is None else marginal_prediction
    steps = []
    for t in range(1, cfg.horizon + 1):
        previous, current = cfg.marginal(t - 1), cfg.marginal(t)
        reset_q0, reset_q1 = pi - pi * lam**t, pi + (1 - pi) * lam**t
        risks = {
            "stateful_markov": mixture(
                previous, expected_loss(cfg.p01, cfg.p01), expected_loss(cfg.p11, cfg.p11)
            ),
            "memoryless_marginal": expected_loss(current, marginal_prediction),
            "reset_each_trial": mixture(
                mu0, expected_loss(reset_q0, cfg.p01), expected_loss(reset_q1, cfg.p11)
            ),
            "deranged_memory": mixture(
                previous, expected_loss(current, cfg.p01), expected_loss(current, cfg.p11)
            ),
        }
        steps.append({"t": t, "previous_marginal": previous, "target_marginal": current, "arms": risks})
    average = {
        a: {m: math.fsum(s["arms"][a][m] for s in steps) / cfg.horizon for m in METRICS}
        for a in ARMS
    }
    return {
        "parameters": {
            "p11": cfg.p11, "p01": cfg.p01, "initial_one_probability": mu0,
            "horizon": cfg.horizon, "stationary_one_probability": pi, "eigenvalue": lam,
            "pooled_target_probability": pooled, "memoryless_prediction": marginal_prediction,
        },
        "average": average,
        "per_step": steps,
    }


def stationary_closed_form(process):
    """Independent compact formulas under mu0=pi, averaging t=1,...,H."""
    pi, lam, h = process.stationary, process.eigenvalue, process.horizon
    v = pi * (1 - pi)
    logit = lambda p: math.log(p) - math.log1p(-p)
    delta_logit = logit(process.p11) - logit(process.p01)
    decay_factor = (1 - lam**h) / (h * (1 - lam))
    state_ce = mixture(pi, expected_loss(process.p01, process.p01), expected_loss(process.p11, process.p11))["bce_nats"]
    deranged_ce = mixture(pi, expected_loss(pi, process.p01), expected_loss(pi, process.p11))["bce_nats"]
    averages = {
        "stateful_markov": {"brier": v * (1 - lam**2), "bce_nats": state_ce},
        "memoryless_marginal": {"brier": v, "bce_nats": expected_loss(pi, pi)["bce_nats"]},
        "reset_each_trial": {
            "brier": v * (1 + lam**2) - 2 * v * lam**2 * decay_factor,
            "bce_nats": deranged_ce - v * lam * delta_logit * decay_factor,
        },
        "deranged_memory": {"brier": v * (1 + lam**2), "bce_nats": deranged_ce},
    }
    reset_by_time = [
        {"t": t, "brier": v * (1 + lam**2 - 2 * lam ** (t + 1)),
         "bce_nats": deranged_ce - v * lam**t * delta_logit}
        for t in range(1, h + 1)
    ]
    return {"average": averages, "reset_per_step": reset_by_time,
            "decay_factor": decay_factor, "logit_difference": delta_logit}


def enumerate_paths(process):
    """Enumerate every binary path; no eigenvalue or stationary formula used."""
    weighted_paths = []
    for sequence in itertools.product((0, 1), repeat=process.horizon + 1):
        mass = process.initial_one_probability if sequence[0] else 1 - process.initial_one_probability
        for previous, current in zip(sequence, sequence[1:]):
            next_one = process.p11 if previous else process.p01
            mass *= next_one if current else 1 - next_one
        weighted_paths.append((sequence, mass))
    return weighted_paths


def enumerated_risks(process):
    """Directly score all realized paths and independent donor/target path pairs."""
    paths = enumerate_paths(process)
    h = process.horizon
    pooled = math.fsum(mass * sum(seq[1:]) for seq, mass in paths) / h
    risks = {a: {m: 0.0 for m in METRICS} for a in ARMS}
    score = lambda y, p: {"brier": (y - p) ** 2, "bce_nats": -math.log(p if y else 1 - p)}
    for seq, mass in paths:
        for t in range(1, h + 1):
            forecasts = {
                "stateful_markov": process.p11 if seq[t - 1] else process.p01,
                "memoryless_marginal": pooled,
                "reset_each_trial": process.p11 if seq[0] else process.p01,
            }
            for a, prediction in forecasts.items():
                realized = score(seq[t], prediction)
                for m in METRICS:
                    risks[a][m] += mass * realized[m] / h
        for donor, donor_mass in paths:
            for t in range(1, h + 1):
                prediction = process.p11 if donor[t - 1] else process.p01
                realized = score(seq[t], prediction)
                for m in METRICS:
                    risks["deranged_memory"][m] += mass * donor_mass * realized[m] / h
    return risks


class IndependentMathTests(unittest.TestCase):
    def check_enumeration(self, cfg):
        paths = enumerate_paths(cfg)
        self.assertAlmostEqual(math.fsum(mass for _, mass in paths), 1, places=13)
        exact, enumerated = reference(cfg)["average"], enumerated_risks(cfg)
        for a in ARMS:
            for m in METRICS:
                self.assertAlmostEqual(exact[a][m], enumerated[a][m], places=11, msg=(a, m))

    def test_exhaustive_nonstationary_paths(self):
        self.check_enumeration(Process(horizon=5))

    def test_exhaustive_stationary_paths(self):
        self.check_enumeration(Process(initial_one_probability=Process().stationary, horizon=5))

    def test_exhaustive_negative_eigenvalue_paths(self):
        self.check_enumeration(Process(p11=0.25, p01=0.8, initial_one_probability=0.17, horizon=5))

    def test_stationary_compact_formulas_at_every_time(self):
        cfg = Process(initial_one_probability=Process().stationary)
        general, compact = reference(cfg), stationary_closed_form(cfg)
        for a in ARMS:
            for m in METRICS:
                self.assertAlmostEqual(general["average"][a][m], compact["average"][a][m], places=13)
        for general_step, compact_step in zip(general["per_step"], compact["reset_per_step"]):
            for m in METRICS:
                self.assertAlmostEqual(general_step["arms"]["reset_each_trial"][m], compact_step[m], places=13)

    def test_t1_reset_equals_stateful(self):
        for mu0 in (0.0, 0.17, 0.5, Process().stationary, 1.0):
            step = reference(Process(initial_one_probability=mu0))["per_step"][0]["arms"]
            for m in METRICS:
                self.assertAlmostEqual(step["stateful_markov"][m], step["reset_each_trial"][m], places=14)

    def test_marginal_eigenvalue_formula_matches_forward_recursion(self):
        cfg = Process()
        probability = cfg.initial_one_probability
        for t in range(cfg.horizon + 1):
            self.assertAlmostEqual(probability, cfg.marginal(t), places=14)
            probability = probability * cfg.p11 + (1 - probability) * cfg.p01

    def test_pooled_marginal_minimizes_constant_forecast_average_risk(self):
        cfg = Process()
        optimal = reference(cfg)["average"]["memoryless_marginal"]
        for p in (0.5, 0.7, cfg.stationary, 0.9):
            alternative = reference(cfg, marginal_prediction=p)["average"]["memoryless_marginal"]
            for m in METRICS:
                self.assertLess(optimal[m], alternative[m])

    def test_general_brier_variance_identities(self):
        cfg = Process()
        mu0, lam = cfg.initial_one_probability, cfg.eigenvalue
        v0 = mu0 * (1 - mu0)
        for s in reference(cfg)["per_step"]:
            t, prev, current = s["t"], s["previous_marginal"], s["target_marginal"]
            deranged = current * (1 - current) + lam**2 * prev * (1 - prev)
            reset = current * (1 - current) + lam**2 * v0 + (cfg.marginal(1) - current)**2 - 2 * v0 * lam ** (t + 1)
            self.assertAlmostEqual(deranged, s["arms"]["deranged_memory"]["brier"], places=14)
            self.assertAlmostEqual(reset, s["arms"]["reset_each_trial"]["brier"], places=14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "math_verification.json")
    args = parser.parse_args()
    tests = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentMathTests))
    if not tests.wasSuccessful():
        return 1
    cfg = Process()
    stationary_cfg = Process(initial_one_probability=cfg.stationary)
    compact = stationary_closed_form(stationary_cfg)
    report = {
        "status": "executed_independent_exact_expectation_and_exhaustive_short_path_checks",
        "scope": "Synthetic Bernoulli Markov forecasting only; no consciousness inference.",
        "tests": {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors)},
        "assumptions": [
            "The transition parameters are known; no finite-training estimation noise is included.",
            "Every step t=1,...,29 is equally weighted; logarithms are natural.",
            "Donor and target trials are independent, with the same initial distribution and transition law.",
            "The default memoryless forecast is the mean target marginal for the fixed horizon.",
            "Reset uses the one-step transition prediction from X0 at every step, not P(Xt=1|X0).",
            "Asymptotic refers to growing independent training trials at fixed horizon; the horizon is 29.",
        ],
        "stationary_initial_distribution": reference(stationary_cfg),
        "stationary_closed_form_cross_check": compact,
        "public_source_initial_distribution": reference(cfg),
        "public_source_fixed_stationary_marginal_comparison": reference(cfg, marginal_prediction=cfg.stationary),
        "pasted_stationary_reset_bce": {
            "claimed": 0.6869,
            "correct": compact["average"]["reset_each_trial"]["bce_nats"],
            "absolute_error": abs(0.6869 - compact["average"]["reset_each_trial"]["bce_nats"]),
            "correct_rounded_four_decimals": round(compact["average"]["reset_each_trial"]["bce_nats"], 4),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    for key in ("stationary_initial_distribution", "public_source_initial_distribution"):
        print(key)
        for arm in ARMS:
            risk = report[key]["average"][arm]
            print(f"  {arm:22s} Brier={risk['brier']:.12f} BCE={risk['bce_nats']:.12f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
