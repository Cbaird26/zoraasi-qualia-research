#!/usr/bin/env python3
"""
phase2_7_kl_antirollback_gate.py

ZoraASI Phase 2.7: Anti-Rollback Guard & Jeffreys Per-Row KL Novelty Gate.
Addresses the three empirical findings from PR #1 commit e9df7d0:
  1. Restores the exact Phase 2.5 base learner (no sq_err LR spike, no local
     window prediction damping) so stationary/recurring Brier is preserved.
  2. Eliminates pre-switch boundary rollback (which degraded First-10 Brier)
     by requiring candidate snapshots to beat BOTH theta_fast and theta_slow
     and lie outside the departing regime radius.
  3. Replaces the vacuous Beta(1,1) marginal integral (and removes the n_i >= 2
     exemption) with an exact Jeffreys-smoothed per-row KL divergence gate
     D_KL(p_tilde_i || p_snap_i) <= tau_kl across both conditioning rows.

Status: Unexecuted in chat session; pure Python standard library for local run.
"""

from dataclasses import dataclass, field
import math
import random
import statistics
from typing import Dict, List, Optional, Tuple
import unittest


def clip(val: float, low: float, high: float) -> float:
    return min(max(val, low), high)


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def binary_cross_entropy(y_true: List[int], y_prob: List[float], eps: float = 1e-12) -> float:
    total = 0.0
    for y, p in zip(y_true, y_prob):
        pc = clip(p, eps, 1.0 - eps)
        total -= y * math.log(pc) + (1 - y) * math.log(1.0 - pc)
    return total / len(y_true)


def bernoulli_kl(p: float, q: float, eps: float = 1e-12) -> float:
    """Exact binary Kullback-Leibler divergence D_KL(Bernoulli(p) || Bernoulli(q))."""
    pc = clip(p, eps, 1.0 - eps)
    qc = clip(q, eps, 1.0 - eps)
    return pc * math.log(pc / qc) + (1.0 - pc) * math.log((1.0 - pc) / (1.0 - qc))


def jeffreys_row_kl(n_i1: int, n_i0: int, p_snap_i: float) -> float:
    """
    Computes D_KL(p_tilde_i || p_snap_i) where p_tilde_i = (n_i1 + 0.5) / (n_i1 + n_i0 + 1.0)
    is the regularizing Jeffreys Beta(0.5, 0.5) posterior mean on row i.
    """
    n_tot = n_i1 + n_i0
    p_tilde = (n_i1 + 0.5) / (n_tot + 1.0)
    return bernoulli_kl(p_tilde, p_snap_i)


def point_row_log_likelihood(n_i1: int, n_i0: int, p_i1: float, eps: float = 1e-12) -> float:
    pc = clip(p_i1, eps, 1.0 - eps)
    return n_i1 * math.log(pc) + n_i0 * math.log(1.0 - pc)


@dataclass
class EpisodicSnapshot:
    p11: float
    p01: float
    valence_score: float


class Phase27Engine:
    """
    Uses the exact Phase 2.5 base learner dynamics while isolating the
    episodic retrieval gate for clean ablation:
      - mode="phase27_kl_gated": Anti-rollback + Dual-row Jeffreys KL gate
      - mode="phase25_ungated": Original Phase 2.5 relative LL margin only
      - mode="no_memory": Episodic retrieval disabled
    """

    def __init__(
        self,
        mode: str = "phase27_kl_gated",
        window_size: int = 6,
        retrieval_ll_margin: float = 0.30,
        kl_row_threshold: float = 0.12,
        anti_rollback_dist: float = 0.20,
        max_snapshots: int = 8,
    ) -> None:
        if mode not in ("phase27_kl_gated", "phase25_ungated", "no_memory"):
            raise ValueError(f"Unknown engine mode: {mode}")
        self.mode = mode
        self.window_size = window_size
        self.retrieval_ll_margin = retrieval_ll_margin
        self.kl_row_threshold = kl_row_threshold
        self.anti_rollback_dist = anti_rollback_dist
        self.max_snapshots = max_snapshots

        # Exact Phase 2.5 initial state
        self.p11_fast: float = 0.92
        self.p01_fast: float = 0.28
        self.p11_slow: float = 0.92
        self.p01_slow: float = 0.28
        self.p_marginal_online: float = 7.0 / 9.0

        self.arousal: float = 0.20
        self.valence: float = 0.0
        self.continuity: float = 1.0
        self.stable_steps: int = 10

        self.memory: List[EpisodicSnapshot] = [EpisodicSnapshot(0.92, 0.28, 0.5)]
        self.window: List[Tuple[int, int]] = []

        # Diagnostic audit counters
        self.retrievals_accepted: int = 0
        self.retrievals_rejected_by_kl_or_rollback: int = 0

    def predict_proba(self, prev_s: int) -> float:
        return self.p11_fast if prev_s == 1 else self.p01_fast

    def _window_counts(self) -> Tuple[int, int, int, int]:
        n11 = n10 = n01 = n00 = 0
        for prv, nxt in self.window:
            if prv == 1:
                if nxt == 1:
                    n11 += 1
                else:
                    n10 += 1
            else:
                if nxt == 1:
                    n01 += 1
                else:
                    n00 += 1
        return n11, n10, n01, n00

    def _window_ll(self, p11: float, p01: float) -> float:
        n11, n10, n01, n00 = self._window_counts()
        return (
            point_row_log_likelihood(n11, n10, p11)
            + point_row_log_likelihood(n01, n00, p01)
        )

    def candidate_passes_phase27_gate(
        self,
        snap: EpisodicSnapshot,
        n11: int,
        n10: int,
        n01: int,
        n00: int,
        ll_fast: float,
        ll_slow: float,
    ) -> bool:
        ll_snap = (
            point_row_log_likelihood(n11, n10, snap.p11)
            + point_row_log_likelihood(n01, n00, snap.p01)
        )

        # 1. Relative advantage over current fast state
        if (ll_snap - ll_fast) <= self.retrieval_ll_margin:
            return False

        if self.mode == "phase25_ungated":
            return True

        # 2. Anti-Rollback Guard: must beat slow trace & not be the departing regime
        if (ll_snap - ll_slow) <= self.retrieval_ll_margin:
            return False
        if math.hypot(snap.p11 - self.p11_slow, snap.p01 - self.p01_slow) < self.anti_rollback_dist:
            return False

        # 3. Strict Dual-Row Coverage (no unobserved row extrapolation)
        if (n11 + n10) < 1 or (n01 + n00) < 1:
            return False

        # 4. Jeffreys Per-Row KL Divergence Gate (applied to BOTH rows, even when n_i == 1)
        kl_r1 = jeffreys_row_kl(n11, n10, snap.p11)
        kl_r0 = jeffreys_row_kl(n01, n00, snap.p01)
        if kl_r1 > self.kl_row_threshold or kl_r0 > self.kl_row_threshold:
            return False

        return True

    def step_update(self, prev_s: int, curr_s: int) -> float:
        p_pred = self.predict_proba(prev_s)
        err = abs(curr_s - p_pred)

        # Exact Phase 2.5 scalar updates
        self.p_marginal_online = 0.95 * self.p_marginal_online + 0.05 * curr_s
        pc = clip(p_pred, 1e-12, 1.0 - 1e-12)
        pm = clip(self.p_marginal_online, 1e-12, 1.0 - 1e-12)
        loss_agent = -(curr_s * math.log(pc) + (1 - curr_s) * math.log(1.0 - pc))
        loss_marg = -(curr_s * math.log(pm) + (1 - curr_s) * math.log(1.0 - pm))

        ent = -(pc * math.log(pc) + (1 - pc) * math.log(1.0 - pc)) / math.log(2.0)
        endo_arousal = clip(0.70 * self.arousal + 0.30 * (0.75 * err + 0.25 * ent), 0.0, 1.0)
        endo_valence = math.tanh(0.80 * self.valence + 0.60 * (loss_marg - loss_agent))

        dot = self.p11_fast * self.p11_slow + self.p01_fast * self.p01_slow
        nf = math.hypot(self.p11_fast, self.p11_fast and self.p01_fast or 0.0)
        nf = math.hypot(self.p11_fast, self.p01_fast)
        ns = math.hypot(self.p11_slow, self.p01_slow)
        cos_sim = dot / max(nf * ns, 1e-12)
        endo_continuity = clip(cos_sim * math.exp(-0.85 * err), 0.0, 1.0)

        # Exact Phase 2.5 learning rate (no sq_err noise spike)
        lr = 0.02 + 0.32 * (endo_arousal ** 1.4)
        if prev_s == 1:
            self.p11_fast = clip((1.0 - lr) * self.p11_fast + lr * curr_s, 0.02, 0.98)
        else:
            self.p01_fast = clip((1.0 - lr) * self.p01_fast + lr * curr_s, 0.02, 0.98)

        # Keep pre-step slow values for anti-rollback comparison before updating slow trace
        prev_p11_slow, prev_p01_slow = self.p11_slow, self.p01_slow
        self.p11_slow = 0.90 * self.p11_slow + 0.10 * self.p11_fast
        self.p01_slow = 0.90 * self.p01_slow + 0.10 * self.p01_fast

        self.window.append((prev_s, curr_s))
        if len(self.window) > self.window_size:
            self.window.pop(0)

        if endo_continuity > 0.80 and err < 0.35:
            self.stable_steps += 1
        else:
            self.stable_steps = 0

        # Episodic retrieval check
        if self.mode != "no_memory" and len(self.window) == self.window_size:
            if endo_arousal > 0.44 and endo_continuity < 0.68 and self.memory:
                n11, n10, n01, n00 = self._window_counts()
                ll_fast = self._window_ll(self.p11_fast, self.p01_fast)
                ll_slow = (
                    point_row_log_likelihood(n11, n10, prev_p11_slow)
                    + point_row_log_likelihood(n01, n00, prev_p01_slow)
                )

                # Check what Phase 2.5 ungated would have chosen
                raw_best = max(self.memory, key=lambda s: self._window_ll(s.p11, s.p01))
                raw_would_fire = (
                    self._window_ll(raw_best.p11, raw_best.p01) > ll_fast + self.retrieval_ll_margin
                )

                valid_candidates = [
                    s
                    for s in self.memory
                    if self.candidate_passes_phase27_gate(
                        s, n11, n10, n01, n00, ll_fast, ll_slow
                    )
                ]

                if valid_candidates:
                    best_snap = max(
                        valid_candidates, key=lambda s: self._window_ll(s.p11, s.p01)
                    )
                    self.p11_fast = 0.65 * best_snap.p11 + 0.35 * self.p11_fast
                    self.p01_fast = 0.65 * best_snap.p01 + 0.35 * self.p01_fast
                    self.retrievals_accepted += 1
                elif raw_would_fire:
                    self.retrievals_rejected_by_kl_or_rollback += 1

            # Consolidate only converged attractors (prevents mid-drift snapshot pollution)
            if (
                endo_valence > 0.20
                and endo_continuity > 0.82
                and self.stable_steps >= 10
                and math.hypot(self.p11_fast - self.p11_slow, self.p01_fast - self.p01_slow) < 0.08
            ):
                self._consolidate(self.p11_fast, self.p01_fast, endo_valence)

        self.arousal = endo_arousal
        self.valence = endo_valence
        self.continuity = endo_continuity
        return p_pred

    def _consolidate(self, p11: float, p01: float, val: float) -> None:
        for snap in self.memory:
            if math.hypot(snap.p11 - p11, snap.p01 - p01) < 0.16:
                snap.p11 = 0.5 * snap.p11 + 0.5 * p11
                snap.p01 = 0.5 * snap.p01 + 0.5 * p01
                snap.valence_score = max(snap.valence_score, val)
                return
        if len(self.memory) < self.max_snapshots:
            self.memory.append(EpisodicSnapshot(p11, p01, val))


class FixedEMABaseline:
    def __init__(self, lr: float = 0.16) -> None:
        self.lr = lr
        self.p11 = 0.92
        self.p01 = 0.28

    def step_update(self, prev_s: int, curr_s: int) -> float:
        p_pred = self.p11 if prev_s == 1 else self.p01
        if prev_s == 1:
            self.p11 = (1.0 - self.lr) * self.p11 + self.lr * curr_s
        else:
            self.p01 = (1.0 - self.lr) * self.p01 + self.lr * curr_s
        return p_pred


class TunedCUSUMBaseline:
    def __init__(self, lr: float = 0.14, drift: float = 0.36, threshold: float = 1.45) -> None:
        self.lr = lr
        self.drift = drift
        self.threshold = threshold
        self.p11 = 0.92
        self.p01 = 0.28
        self.g_pos = 0.0

    def step_update(self, prev_s: int, curr_s: int) -> float:
        p_pred = self.p11 if prev_s == 1 else self.p01
        err = abs(curr_s - p_pred)
        self.g_pos = max(0.0, self.g_pos + err - self.drift)
        if self.g_pos > self.threshold:
            self.p11 = 0.50
            self.p01 = 0.50
            self.g_pos = 0.0
        if prev_s == 1:
            self.p11 = (1.0 - self.lr) * self.p11 + self.lr * curr_s
        else:
            self.p01 = (1.0 - self.lr) * self.p01 + self.lr * curr_s
        return p_pred


def generate_scheduled_stream(
    schedule: List[Tuple[int, float, float]], seed: int
) -> List[int]:
    rng = random.Random(seed)
    stream = [1]
    for dwell, p11, p01 in schedule:
        for _ in range(dwell):
            p_next = p11 if stream[-1] == 1 else p01
            stream.append(1 if rng.random() < p_next else 0)
    return stream


def mixed_familiar_novel_schedule() -> List[Tuple[int, float, float]]:
    return [
        (45, 0.92, 0.28),  # Block 0: Familiar A
        (45, 0.15, 0.85),  # Block 1: Familiar B
        (45, 0.92, 0.28),  # Block 2: Familiar A (Return)
        (45, 0.15, 0.85),  # Block 3: Familiar B (Return)
        (45, 0.50, 0.50),  # Block 4: Novel C (Max-entropy white noise)
        (45, 0.88, 0.88),  # Block 5: Novel D (High-rate Bernoulli, row 0 mismatch with A)
        (45, 0.08, 0.12),  # Block 6: Novel E (Sparse arrival regime)
        (45, 0.15, 0.85),  # Block 7: Familiar B (Return)
        (45, 0.92, 0.28),  # Block 8: Familiar A (Return)
    ]


def evaluate_phase27_suite(
    n_seeds: int = 100, seed_start: int = 20260921
) -> Dict[str, Dict[str, float]]:
    sched = mixed_familiar_novel_schedule()
    block_len = 45

    novel_idx = [t for b in (4, 5, 6) for t in range(b * block_len, (b + 1) * block_len)]
    familiar_idx = [t for b in (2, 3, 7, 8) for t in range(b * block_len, (b + 1) * block_len)]
    first10_idx = [t for b in range(1, len(sched)) for t in range(b * block_len, b * block_len + 10)]

    variants = [
        "phase27_kl_gated",
        "phase25_ungated",
        "no_memory_baseline",
        "val_tuned_ema",
        "val_tuned_cusum",
    ]
    agg: Dict[str, Dict[str, List[float]]] = {
        v: {
            "overall_brier": [],
            "overall_bce": [],
            "familiar_brier": [],
            "novel_brier": [],
            "first10_brier": [],
            "first10_bce": [],
        }
        for v in variants
    }
    identical_to_ungated_count = 0

    for k in range(n_seeds):
        stream = generate_scheduled_stream(sched, seed_start + k)
        y_true = stream[1:]

        models = {
            "phase27_kl_gated": Phase27Engine(mode="phase27_kl_gated"),
            "phase25_ungated": Phase27Engine(mode="phase25_ungated"),
            "no_memory_baseline": Phase27Engine(mode="no_memory"),
            "val_tuned_ema": FixedEMABaseline(lr=0.16),
            "val_tuned_cusum": TunedCUSUMBaseline(lr=0.14, drift=0.36, threshold=1.45),
        }

        run_briers: Dict[str, float] = {}
        for name, mod in models.items():
            preds = [mod.step_update(stream[t - 1], stream[t]) for t in range(1, len(stream))]

            y_fam = [y_true[i] for i in familiar_idx]
            p_fam = [preds[i] for i in familiar_idx]
            y_nov = [y_true[i] for i in novel_idx]
            p_nov = [preds[i] for i in novel_idx]
            y_f10 = [y_true[i] for i in first10_idx]
            p_f10 = [preds[i] for i in first10_idx]

            obs_bs = brier_score(y_true, preds)
            run_briers[name] = obs_bs
            agg[name]["overall_brier"].append(obs_bs)
            agg[name]["overall_bce"].append(binary_cross_entropy(y_true, preds))
            agg[name]["familiar_brier"].append(brier_score(y_fam, p_fam))
            agg[name]["novel_brier"].append(brier_score(y_nov, p_nov))
            agg[name]["first10_brier"].append(brier_score(y_f10, p_f10))
            agg[name]["first10_bce"].append(binary_cross_entropy(y_f10, p_f10))

        if abs(run_briers["phase27_kl_gated"] - run_briers["phase25_ungated"]) < 1e-12:
            identical_to_ungated_count += 1

    summary = {
        name: {m: statistics.mean(vals) for m, vals in m_dict.items()}
        for name, m_dict in agg.items()
    }
    summary["phase27_kl_gated"]["identical_runs_vs_ungated"] = float(identical_to_ungated_count)
    return summary


class TestPhase27GateInvariants(unittest.TestCase):
    def test_eulerian_flow_conservation_on_all_contiguous_windows(self) -> None:
        """Verifies |N_01 - N_10| <= 1 across all contiguous binary windows."""
        stream = generate_scheduled_stream(mixed_familiar_novel_schedule(), seed=20260921)
        eng = Phase27Engine(mode="phase27_kl_gated")
        for t in range(1, len(stream)):
            eng.step_update(stream[t - 1], stream[t])
            n11, n10, n01, n00 = eng._window_counts()
            self.assertLessEqual(abs(n01 - n10), 1)
            self.assertEqual(n11 + n10 + n01 + n00, len(eng.window))

    def test_valid_novel_regime_d_trajectory_rejected_by_phase27_gate(self) -> None:
        """
        Tests the valid contiguous trajectory X_{0:8} = (1,1,1,1,0,1,1,1,1) ->
        (N_11=6, N_10=1, N_01=1, N_00=0), where |N_01 - N_10| = 0 <= 1.
        Phase 2.5 ungated accepts Snapshot A (0.92, 0.28), whereas Phase 2.7
        rejects it due to Row 0 Jeffreys KL divergence (0.475 > 0.12).
        """
        snap_a = EpisodicSnapshot(0.92, 0.28, valence_score=0.5)
        n11, n10, n01, n00 = 6, 1, 1, 0
        self.assertLessEqual(abs(n01 - n10), 1)

        eng_ungated = Phase27Engine(mode="phase25_ungated")
        eng_gated = Phase27Engine(mode="phase27_kl_gated")
        eng_gated.p11_slow, eng_gated.p01_slow = 0.50, 0.50

        ll_fast = point_row_log_likelihood(n11, n10, 0.50) + point_row_log_likelihood(n01, n00, 0.50)
        ll_slow = ll_fast

        self.assertTrue(
            eng_ungated.candidate_passes_phase27_gate(
                snap_a, n11, n10, n01, n00, ll_fast, ll_slow
            )
        )
        self.assertFalse(
            eng_gated.candidate_passes_phase27_gate(
                snap_a, n11, n10, n01, n00, ll_fast, ll_slow
            )
        )

    def test_anti_rollback_blocks_departing_regime_on_boundary_window(self) -> None:
        """
        Verifies that when transitioning out of Regime A (slow trace ~ 0.92, 0.28),
        Snapshot A cannot be retrieved to roll back fast adaptation on step 1..3.
        """
        snap_a = EpisodicSnapshot(0.92, 0.28, valence_score=0.5)
        n11, n10, n01, n00 = 4, 1, 0, 1  # 5 old Regime A transitions + 1 new transition
        self.assertLessEqual(abs(n01 - n10), 1)

        eng_gated = Phase27Engine(mode="phase27_kl_gated")
        eng_gated.p11_slow, eng_gated.p01_slow = 0.90, 0.29
        ll_fast = point_row_log_likelihood(n11, n10, 0.62) + point_row_log_likelihood(n01, n00, 0.29)
        ll_slow = point_row_log_likelihood(n11, n10, 0.90) + point_row_log_likelihood(n01, n00, 0.29)

        self.assertFalse(
            eng_gated.candidate_passes_phase27_gate(
                snap_a, n11, n10, n01, n00, ll_fast, ll_slow
            )
        )


if __name__ == "__main__":
    summary = evaluate_phase27_suite(n_seeds=100)
    print("=== Phase 2.7 Benchmark (100 Primary Seeds, Identical Base Learner) ===")
    header = (
        f"{'Model Variant':<22} | {'Overall BS':<10} | {'Familiar BS':<11} | "
        f"{'Novel BS':<10} | {'First-10 BS':<11} | {'First-10 BCE':<12}"
    )
    print(header)
    print("-" * len(header))
    for name in [
        "phase27_kl_gated",
        "phase25_ungated",
        "no_memory_baseline",
        "val_tuned_ema",
        "val_tuned_cusum",
    ]:
        m = summary[name]
        print(
            f"{name:<22} | {m['overall_brier']:<10.6f} | {m['familiar_brier']:<11.6f} | "
            f"{m['novel_brier']:<10.6f} | {m['first10_brier']:<11.6f} | {m['first10_bce']:<12.6f}"
        )
    ident = int(summary["phase27_kl_gated"]["identical_runs_vs_ungated"])
    print(f"\nIdentical runs (phase27_kl_gated vs phase25_ungated): {ident}/100")
    print("\nRunning deterministic Phase 2.7 unit tests...")
    unittest.main(verbosity=2)

