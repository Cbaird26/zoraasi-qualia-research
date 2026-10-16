#!/usr/bin/env python3
"""
phase2_3_protocol_reproduction.py

Corrected ZoraASI Phase 2.3 baseline experiment implementing:
  - Correction A: Beta(1,1) / Laplace smoothing on transition & marginal estimators
  - Correction B: Fixed cross-trial, same-time-step derangement control
  - Correction C: Fourth experimental arm (reset-each-trial initial state X_{i,0})
  - Correction D: Separation of deterministic software invariants from
                  multi-seed stochastic performance distributions.

Verification status: Unexecuted in chat session; ready for local execution.
"""

from dataclasses import dataclass
import math
import random
import statistics
from typing import Dict, List, Tuple
import unittest


@dataclass(frozen=True)
class Phase23Config:
    seed: int = 20260921
    p11: float = 0.92
    p01: float = 0.28
    n_train_trials: int = 24
    n_test_trials: int = 12
    seq_length: int = 30
    laplace_alpha: float = 1.0  # Beta(1,1) prior smoothing
    eps: float = 1e-12


def stationary_prob(p11: float, p01: float) -> float:
    return p01 / (p01 + (1.0 - p11))


def generate_trials(
    n_trials: int, seq_length: int, p11: float, p01: float, rng: random.Random
) -> List[List[int]]:
    pi_1 = stationary_prob(p11, p01)
    trials: List[List[int]] = []
    for _ in range(n_trials):
        x0 = 1 if rng.random() < pi_1 else 0
        seq = [x0]
        for _ in range(1, seq_length):
            p_next = p11 if seq[-1] == 1 else p01
            seq.append(1 if rng.random() < p_next else 0)
        trials.append(seq)
    return trials


def fixed_cyclic_derangement(n_trials: int) -> List[int]:
    """Returns a deterministic fixed-point-free permutation sigma(i) != i."""
    if n_trials < 2:
        raise ValueError("Derangement requires at least 2 trials.")
    return [(i + 1) % n_trials for i in range(n_trials)]


@dataclass
class SmoothedMarkovEstimators:
    p11_hat: float
    p01_hat: float
    p_marginal_hat: float
    n11: int
    n1_total: int
    n01: int
    n0_total: int

    @classmethod
    def fit(cls, trials: List[List[int]], alpha: float = 1.0) -> "SmoothedMarkovEstimators":
        n11 = n1_total = n01 = n0_total = 0
        for seq in trials:
            for t in range(1, len(seq)):
                prev_s, next_s = seq[t - 1], seq[t]
                if prev_s == 1:
                    n1_total += 1
                    if next_s == 1:
                        n11 += 1
                else:
                    n0_total += 1
                    if next_s == 1:
                        n01 += 1

        total_transitions = n1_total + n0_total
        total_arrivals = n11 + n01

        # Beta(alpha, alpha) Laplace-smoothed estimators
        p11_hat = (n11 + alpha) / (n1_total + 2.0 * alpha)
        p01_hat = (n01 + alpha) / (n0_total + 2.0 * alpha)
        p_marginal_hat = (total_arrivals + alpha) / (total_transitions + 2.0 * alpha)

        return cls(
            p11_hat=p11_hat,
            p01_hat=p01_hat,
            p_marginal_hat=p_marginal_hat,
            n11=n11,
            n1_total=n1_total,
            n01=n01,
            n0_total=n0_total,
        )

    def predict_from_state(self, state: int) -> float:
        return self.p11_hat if state == 1 else self.p01_hat


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def binary_cross_entropy(y_true: List[int], y_prob: List[float], eps: float = 1e-12) -> float:
    loss = 0.0
    for y, p in zip(y_true, y_prob):
        pc = min(max(p, eps), 1.0 - eps)
        loss -= y * math.log(pc) + (1 - y) * math.log(1.0 - pc)
    return loss / len(y_true)


def evaluate_four_arms(
    test_trials: List[List[int]], model: SmoothedMarkovEstimators, eps: float = 1e-12
) -> Dict[str, Dict[str, float]]:
    n_trials = len(test_trials)
    seq_len = len(test_trials[0])
    sigma = fixed_cyclic_derangement(n_trials)

    y_true: List[int] = []
    preds_stateful: List[float] = []
    preds_memoryless: List[float] = []
    preds_reset_trial: List[float] = []
    preds_deranged: List[float] = []

    for i in range(n_trials):
        initial_state = test_trials[i][0]
        deranged_trial_idx = sigma[i]
        for t in range(1, seq_len):
            y_true.append(test_trials[i][t])

            # Arm 1: True prior state X_{i, t-1}
            preds_stateful.append(model.predict_from_state(test_trials[i][t - 1]))

            # Arm 2: Memoryless marginal rate
            preds_memoryless.append(model.p_marginal_hat)

            # Arm 3: Reset-each-trial comparator anchored to initial state X_{i, 0}
            preds_reset_trial.append(model.predict_from_state(initial_state))

            # Arm 4: Cross-trial, same-time-step derangement X_{sigma(i), t-1}
            deranged_prev_state = test_trials[deranged_trial_idx][t - 1]
            preds_deranged.append(model.predict_from_state(deranged_prev_state))

    arms = {
        "stateful_markov": preds_stateful,
        "memoryless_marginal": preds_memoryless,
        "reset_each_trial": preds_reset_trial,
        "deranged_memory": preds_deranged,
    }
    return {
        name: {
            "brier": brier_score(y_true, preds),
            "bce": binary_cross_entropy(y_true, preds, eps),
        }
        for name, preds in arms.items()
    }


def run_single_seed(cfg: Phase23Config = Phase23Config()) -> Dict[str, object]:
    rng = random.Random(cfg.seed)
    train_trials = generate_trials(cfg.n_train_trials, cfg.seq_length, cfg.p11, cfg.p01, rng)
    test_trials = generate_trials(cfg.n_test_trials, cfg.seq_length, cfg.p11, cfg.p01, rng)
    model = SmoothedMarkovEstimators.fit(train_trials, alpha=cfg.laplace_alpha)
    metrics = evaluate_four_arms(test_trials, model, eps=cfg.eps)
    return {"model": model, "metrics": metrics, "test_trials": test_trials}


def run_multi_seed_benchmark(
    n_seeds: int = 200, base_seed: int = 20260921
) -> Dict[str, Dict[str, float]]:
    """Evaluates mean, SD, and 95% CI across multiple seeds to separate stochastic noise."""
    arm_names = [
        "stateful_markov",
        "memoryless_marginal",
        "reset_each_trial",
        "deranged_memory",
    ]
    brier_samples: Dict[str, List[float]] = {k: [] for k in arm_names}
    bce_samples: Dict[str, List[float]] = {k: [] for k in arm_names}

    for k in range(n_seeds):
        cfg = Phase23Config(seed=base_seed + k)
        out = run_single_seed(cfg)
        metrics = out["metrics"]  # type: ignore[assignment]
        for arm in arm_names:
            brier_samples[arm].append(metrics[arm]["brier"])
            bce_samples[arm].append(metrics[arm]["bce"])

    summary: Dict[str, Dict[str, float]] = {}
    for arm in arm_names:
        bs_mean = statistics.mean(brier_samples[arm])
        bs_sd = statistics.stdev(brier_samples[arm])
        bce_mean = statistics.mean(bce_samples[arm])
        bce_sd = statistics.stdev(bce_samples[arm])
        summary[arm] = {
            "brier_mean": bs_mean,
            "brier_ci95": 1.96 * bs_sd / math.sqrt(n_seeds),
            "bce_mean": bce_mean,
            "bce_ci95": 1.96 * bce_sd / math.sqrt(n_seeds),
        }
    return summary


class TestPhase23DeterministicInvariants(unittest.TestCase):
    """Tests strict mathematical and software invariants independent of finite-sample luck."""

    def setUp(self) -> None:
        self.cfg = Phase23Config()
        self.run_data = run_single_seed(self.cfg)
        self.model: SmoothedMarkovEstimators = self.run_data["model"]  # type: ignore[assignment]

    def test_exact_laplace_smoothing_formula(self) -> None:
        expected_p11 = (self.model.n11 + 1.0) / (self.model.n1_total + 2.0)
        expected_p01 = (self.model.n01 + 1.0) / (self.model.n0_total + 2.0)
        self.assertAlmostEqual(self.model.p11_hat, expected_p11, places=12)
        self.assertAlmostEqual(self.model.p01_hat, expected_p01, places=12)

    def test_derangement_has_no_fixed_points(self) -> None:
        sigma = fixed_cyclic_derangement(self.cfg.n_test_trials)
        self.assertEqual(len(sigma), self.cfg.n_test_trials)
        self.assertEqual(sorted(sigma), list(range(self.cfg.n_test_trials)))
        for i, mapped in enumerate(sigma):
            self.assertNotEqual(i, mapped)

    def test_transition_counts_and_four_arms_present(self) -> None:
        self.assertEqual(self.model.n1_total + self.model.n0_total, 24 * 29)
        metrics: Dict[str, Dict[str, float]] = self.run_data["metrics"]  # type: ignore[assignment]
        self.assertEqual(
            set(metrics.keys()),
            {
                "stateful_markov",
                "memoryless_marginal",
                "reset_each_trial",
                "deranged_memory",
            },
        )

    def test_deterministic_seed_reproducibility(self) -> None:
        repeat = run_single_seed(self.cfg)
        self.assertEqual(self.run_data["metrics"], repeat["metrics"])


if __name__ == "__main__":
    single = run_single_seed()
    m: SmoothedMarkovEstimators = single["model"]  # type: ignore[assignment]
    metrics: Dict[str, Dict[str, float]] = single["metrics"]  # type: ignore[assignment]

    print("=== Phase 2.3 Protocol Reproduction (Seed: 20260921) ===")
    print(
        f"Laplace-Smoothed Estimators -> p11_hat: {m.p11_hat:.4f} "
        f"({m.n11}+1)/({m.n1_total}+2), "
        f"p01_hat: {m.p01_hat:.4f} ({m.n01}+1)/({m.n0_total}+2), "
        f"p_marginal_hat: {m.p_marginal_hat:.4f}\n"
    )
    print(f"{'Experimental Arm':<24} | {'Brier Score':<12} | {'BCE (nats)':<12}")
    print("-" * 54)
    for arm_key in [
        "stateful_markov",
        "memoryless_marginal",
        "reset_each_trial",
        "deranged_memory",
    ]:
        print(
            f"{arm_key:<24} | {metrics[arm_key]['brier']:<12.6f} | "
            f"{metrics[arm_key]['bce']:<12.6f}"
        )

    print("\n=== Multi-Seed Benchmark (N=200 Seeds, Mean ± 95% CI) ===")
    multi = run_multi_seed_benchmark(n_seeds=200)
    for arm_key, stats in multi.items():
        print(
            f"{arm_key:<24} | Brier: {stats['brier_mean']:.4f} ± {stats['brier_ci95']:.4f} "
            f"| BCE: {stats['bce_mean']:.4f} ± {stats['bce_ci95']:.4f}"
        )

    print("\nRunning deterministic software invariant unit tests...")
    unittest.main(verbosity=2)

