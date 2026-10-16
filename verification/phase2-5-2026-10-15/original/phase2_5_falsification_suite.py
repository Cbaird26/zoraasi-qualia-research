#!/usr/bin/env python3
"""
phase2_5_falsification_suite.py

Addresses the three falsifiable follow-up milestones identified in PR #1 (b25e8c2):
  1. Survival against tuned adaptive baselines (Grid-Searched EMA & CUSUM Reset)
  2. Generalization across familiar vs. unfamiliar (novel) Markov regimes
  3. Robustness against desynchronized cross-stream scalar transfer controls
     and initial-state distribution shifts.

Status: Unexecuted in chat session; pure Python standard library for local run.
"""

from dataclasses import dataclass, field
import math
import random
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


@dataclass
class ScalarTrace:
    """Stores per-step scalar states for cross-stream donor transfer controls."""
    arousal: List[float] = field(default_factory=list)
    valence: List[float] = field(default_factory=list)
    continuity: List[float] = field(default_factory=list)


@dataclass
class EpisodicSnapshot:
    p11: float
    p01: float
    valence_score: float


class RefinedQualiaEngineV1_1:
    """
    Refined Phase 2.5 engine:
      - Removes static-marginal output blending (which degraded Brier in PR #1)
      - Uses continuity & arousal strictly for plasticity modulation and
        likelihood-gated episodic snapshot retrieval
      - Supports injecting donor scalar traces for Experiment 3 controls.
    """

    def __init__(
        self,
        enable_episodic_memory: bool = True,
        enable_affective_plasticity: bool = True,
        fixed_lr: float = 0.12,
        retrieval_ll_margin: float = 0.30,
        max_snapshots: int = 8,
    ) -> None:
        self.enable_episodic_memory = enable_episodic_memory
        self.enable_affective_plasticity = enable_affective_plasticity
        self.fixed_lr = fixed_lr
        self.retrieval_ll_margin = retrieval_ll_margin
        self.max_snapshots = max_snapshots

        self.p11_fast: float = 0.92
        self.p01_fast: float = 0.28
        self.p11_slow: float = 0.92
        self.p01_slow: float = 0.28
        self.p_marginal_online: float = 7.0 / 9.0

        self.arousal: float = 0.20
        self.valence: float = 0.0
        self.continuity: float = 1.0

        self.memory: List[EpisodicSnapshot] = [EpisodicSnapshot(0.92, 0.28, 0.5)]
        self.window: List[Tuple[int, int]] = []
        self.recorded_trace = ScalarTrace()

    def predict_proba(self, prev_s: int) -> float:
        return self.p11_fast if prev_s == 1 else self.p01_fast

    def step_update(
        self,
        prev_s: int,
        curr_s: int,
        donor_scalars: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        p_pred = self.predict_proba(prev_s)
        err = abs(curr_s - p_pred)

        # Online marginal baseline for valence comparison (avoids stale Regime A bias)
        self.p_marginal_online = 0.95 * self.p_marginal_online + 0.05 * curr_s
        pc = clip(p_pred, 1e-12, 1.0 - 1e-12)
        pm = clip(self.p_marginal_online, 1e-12, 1.0 - 1e-12)
        loss_agent = -(curr_s * math.log(pc) + (1 - curr_s) * math.log(1.0 - pc))
        loss_marg = -(curr_s * math.log(pm) + (1 - curr_s) * math.log(1.0 - pm))

        # Compute endogenous scalar states
        ent = -(pc * math.log(pc) + (1 - pc) * math.log(1.0 - pc)) / math.log(2.0)
        endo_arousal = clip(0.70 * self.arousal + 0.30 * (0.75 * err + 0.25 * ent), 0.0, 1.0)
        endo_valence = math.tanh(0.80 * self.valence + 0.60 * (loss_marg - loss_agent))

        dot = self.p11_fast * self.p11_slow + self.p01_fast * self.p01_slow
        nf = math.hypot(self.p11_fast, self.p01_fast)
        ns = math.hypot(self.p11_slow, self.p01_slow)
        cos_sim = dot / max(nf * ns, 1e-12)
        endo_continuity = clip(cos_sim * math.exp(-0.85 * err), 0.0, 1.0)

        # Record endogenous trace before optional donor override
        self.recorded_trace.arousal.append(endo_arousal)
        self.recorded_trace.valence.append(endo_valence)
        self.recorded_trace.continuity.append(endo_continuity)

        if donor_scalars is not None:
            act_arousal, act_valence, act_continuity = donor_scalars
        else:
            act_arousal, act_valence, act_continuity = (
                endo_arousal,
                endo_valence,
                endo_continuity,
            )

        # Plasticity update
        if self.enable_affective_plasticity:
            lr = 0.02 + 0.32 * (act_arousal ** 1.4)
        else:
            lr = self.fixed_lr

        if prev_s == 1:
            self.p11_fast = clip((1.0 - lr) * self.p11_fast + lr * curr_s, 0.02, 0.98)
        else:
            self.p01_fast = clip((1.0 - lr) * self.p01_fast + lr * curr_s, 0.02, 0.98)

        self.p11_slow = 0.90 * self.p11_slow + 0.10 * self.p11_fast
        self.p01_slow = 0.90 * self.p01_slow + 0.10 * self.p01_fast

        # Short transition buffer for likelihood-gated episodic memory
        self.window.append((prev_s, curr_s))
        if len(self.window) > 6:
            self.window.pop(0)

        if self.enable_episodic_memory and len(self.window) == 6:
            if act_arousal > 0.44 and act_continuity < 0.68 and self.memory:
                best_snap = max(self.memory, key=lambda s: self._window_ll(s.p11, s.p01))
                curr_ll = self._window_ll(self.p11_fast, self.p01_fast)
                if self._window_ll(best_snap.p11, best_snap.p01) > curr_ll + self.retrieval_ll_margin:
                    self.p11_fast = 0.65 * best_snap.p11 + 0.35 * self.p11_fast
                    self.p01_fast = 0.65 * best_snap.p01 + 0.35 * self.p01_fast

            if act_valence > 0.20 and act_continuity > 0.82:
                self._consolidate(self.p11_fast, self.p01_fast, act_valence)

        self.arousal = endo_arousal
        self.valence = endo_valence
        self.continuity = endo_continuity
        return p_pred

    def _window_ll(self, p11: float, p01: float) -> float:
        ll = 0.0
        for prv, nxt in self.window:
            p = clip(p11 if prv == 1 else p01, 1e-12, 1.0 - 1e-12)
            ll += nxt * math.log(p) + (1 - nxt) * math.log(1.0 - p)
        return ll

    def _consolidate(self, p11: float, p01: float, val: float) -> None:
        for snap in self.memory:
            if math.hypot(snap.p11 - p11, snap.p01 - p01) < 0.12:
                snap.p11 = 0.5 * snap.p11 + 0.5 * p11
                snap.p01 = 0.5 * snap.p01 + 0.5 * p01
                snap.valence_score = max(snap.valence_score, val)
                return
        if len(self.memory) < self.max_snapshots:
            self.memory.append(EpisodicSnapshot(p11, p01, val))


class PageHinkleyChangepointBaseline:
    """Strong non-affective adaptive baseline: EMA + Page-Hinkley surprise reset."""

    def __init__(self, base_lr: float = 0.10, threshold: float = 1.65) -> None:
        self.base_lr = base_lr
        self.threshold = threshold
        self.p11 = 0.92
        self.p01 = 0.28
        self.cum_err = 0.0
        self.min_cum_err = 0.0

    def step_update(self, prev_s: int, curr_s: int) -> float:
        p_pred = self.p11 if prev_s == 1 else self.p01
        err = abs(curr_s - p_pred)

        # Page-Hinkley statistic for abrupt error drift
        self.cum_err += err - 0.38
        self.min_cum_err = min(self.min_cum_err, self.cum_err)
        if (self.cum_err - self.min_cum_err) > self.threshold:
            # Reset parameters toward uninformative 0.50 prior upon changepoint
            self.p11 = 0.50
            self.p01 = 0.50
            self.cum_err = 0.0
            self.min_cum_err = 0.0

        if prev_s == 1:
            self.p11 = (1.0 - self.base_lr) * self.p11 + self.base_lr * curr_s
        else:
            self.p01 = (1.0 - self.base_lr) * self.p01 + self.base_lr * curr_s
        return p_pred


def generate_scheduled_stream(
    schedule: List[Tuple[int, float, float]], seed: int
) -> List[int]:
    """Generates a stream from an explicit list of (dwell_steps, p11, p01) blocks."""
    rng = random.Random(seed)
    stream = [1]
    for dwell, p11, p01 in schedule:
        for _ in range(dwell):
            p_next = p11 if stream[-1] == 1 else p01
            stream.append(1 if rng.random() < p_next else 0)
    return stream


def run_experiment_1_tuned_baselines(seed: int = 20260921) -> Dict[str, float]:
    """
    Experiment 1: Compares Refined Engine v1.1 against a grid-searched optimal
    fixed-rate EMA and a Page-Hinkley changepoint detector on recurring regimes.
    """
    regimes = [(0.92, 0.28), (0.15, 0.85)]
    schedule = [(45, regimes[b % 2][0], regimes[b % 2][1]) for b in range(10)]
    stream = generate_scheduled_stream(schedule, seed)
    y_true = stream[1:]

    # Grid-search the single best fixed learning rate on this exact stream (oracle EMA)
    best_ema_brier = 1.0
    best_lr = 0.05
    for lr_idx in range(1, 45):
        lr = lr_idx * 0.01
        eng = RefinedQualiaEngineV1_1(
            enable_episodic_memory=False, enable_affective_plasticity=False, fixed_lr=lr
        )
        preds = [eng.step_update(stream[t - 1], stream[t]) for t in range(1, len(stream))]
        bs = brier_score(y_true, preds)
        if bs < best_ema_brier:
            best_ema_brier = bs
            best_lr = lr

    ph_model = PageHinkleyChangepointBaseline()
    ph_preds = [ph_model.step_update(stream[t - 1], stream[t]) for t in range(1, len(stream))]

    full_eng = RefinedQualiaEngineV1_1(enable_episodic_memory=True, enable_affective_plasticity=True)
    full_preds = [full_eng.step_update(stream[t - 1], stream[t]) for t in range(1, len(stream))]

    return {
        "refined_engine_v1_1_brier": brier_score(y_true, full_preds),
        "oracle_tuned_ema_brier": best_ema_brier,
        "oracle_tuned_ema_lr": best_lr,
        "page_hinkley_reset_brier": brier_score(y_true, ph_preds),
    }


def run_experiment_2_unfamiliar_regimes(seed: int = 20260921) -> Dict[str, float]:
    """
    Experiment 2: Alternates between familiar revisited regimes and novel
    unfamiliar regimes to verify that likelihood-gated memory does not
    corrupt adaptation when entering unseen dynamics.
    """
    schedule = [
        (45, 0.92, 0.28),  # Familiar A
        (45, 0.15, 0.85),  # Familiar B
        (45, 0.92, 0.28),  # Familiar A (revisited)
        (45, 0.15, 0.85),  # Familiar B (revisited)
        (45, 0.50, 0.50),  # Unfamiliar C (Maximum-entropy white noise)
        (45, 0.88, 0.88),  # Unfamiliar D (High-rate Bernoulli)
        (45, 0.15, 0.85),  # Familiar B (revisited after novel regimes)
        (45, 0.92, 0.28),  # Familiar A (revisited)
    ]
    stream = generate_scheduled_stream(schedule, seed)
    y_true = stream[1:]

    with_mem = RefinedQualiaEngineV1_1(enable_episodic_memory=True, enable_affective_plasticity=True)
    no_mem = RefinedQualiaEngineV1_1(enable_episodic_memory=False, enable_affective_plasticity=True)

    preds_mem = [with_mem.step_update(stream[t - 1], stream[t]) for t in range(1, len(stream))]
    preds_nomem = [no_mem.step_update(stream[t - 1], stream[t]) for t in range(1, len(stream))]

    # Slice indices for revisited familiar blocks vs unfamiliar blocks
    unfamiliar_slice = slice(4 * 45, 6 * 45)
    revisited_slice = slice(6 * 45, 8 * 45)

    return {
        "overall_with_memory_brier": brier_score(y_true, preds_mem),
        "overall_no_memory_brier": brier_score(y_true, preds_nomem),
        "unfamiliar_blocks_with_memory_brier": brier_score(
            y_true[unfamiliar_slice], preds_mem[unfamiliar_slice]
        ),
        "unfamiliar_blocks_no_memory_brier": brier_score(
            y_true[unfamiliar_slice], preds_nomem[unfamiliar_slice]
        ),
        "revisited_blocks_with_memory_brier": brier_score(
            y_true[revisited_slice], preds_mem[revisited_slice]
        ),
        "revisited_blocks_no_memory_brier": brier_score(
            y_true[revisited_slice], preds_nomem[revisited_slice]
        ),
    }


def run_experiment_3_desynchronized_donor_control(seed: int = 20260921) -> Dict[str, float]:
    """
    Experiment 3: Separates regime timing between recipient and donor streams.
      - Synchronous Donor: shares exact 45-step regime switching clock (PR #1 confound)
      - Desynchronized Donor: switches on an out-of-phase schedule (offset by 22 steps)
    """
    regimes = [(0.92, 0.28), (0.15, 0.85)]
    recipient_sched = [(45, regimes[b % 2][0], regimes[b % 2][1]) for b in range(8)]
    sync_donor_sched = list(recipient_sched)
    # Desynchronized donor switches halfway through every recipient plateau
    desync_donor_sched = [(22, 0.92, 0.28)] + [
        (45, regimes[(b + 1) % 2][0], regimes[(b + 1) % 2][1]) for b in range(7)
    ] + [(23, 0.15, 0.85)]

    rec_stream = generate_scheduled_stream(recipient_sched, seed)
    sync_stream = generate_scheduled_stream(sync_donor_sched, seed + 101)
    desync_stream = generate_scheduled_stream(desync_donor_sched, seed + 202)

    # Extract scalar traces from synchronous and desynchronized donors
    sync_donor = RefinedQualiaEngineV1_1()
    for t in range(1, len(sync_stream)):
        sync_donor.step_update(sync_stream[t - 1], sync_stream[t])

    desync_donor = RefinedQualiaEngineV1_1()
    for t in range(1, len(desync_stream)):
        desync_donor.step_update(desync_stream[t - 1], desync_stream[t])

    # Evaluate recipient under (a) endogenous states, (b) sync donor, (c) desync donor
    eng_endo = RefinedQualiaEngineV1_1()
    eng_sync = RefinedQualiaEngineV1_1()
    eng_desync = RefinedQualiaEngineV1_1()

    preds_endo, preds_sync, preds_desync = [], [], []
    for t in range(1, len(rec_stream)):
        idx = t - 1
        prv, cur = rec_stream[t - 1], rec_stream[t]
        preds_endo.append(eng_endo.step_update(prv, cur))
        preds_sync.append(
            eng_sync.step_update(
                prv,
                cur,
                (
                    sync_donor.recorded_trace.arousal[idx],
                    sync_donor.recorded_trace.valence[idx],
                    sync_donor.recorded_trace.continuity[idx],
                ),
            )
        )
        preds_desync.append(
            eng_desync.step_update(
                prv,
                cur,
                (
                    desync_donor.recorded_trace.arousal[idx],
                    desync_donor.recorded_trace.valence[idx],
                    desync_donor.recorded_trace.continuity[idx],
                ),
            )
        )

    y_true = rec_stream[1:]
    return {
        "endogenous_aligned_brier": brier_score(y_true, preds_endo),
        "synchronous_donor_brier": brier_score(y_true, preds_sync),
        "desynchronized_donor_brier": brier_score(y_true, preds_desync),
    }


class TestPhase25FalsificationSuite(unittest.TestCase):
    def test_endogenous_beats_desynchronized_donor(self) -> None:
        res = run_experiment_3_desynchronized_donor_control()
        self.assertLess(res["endogenous_aligned_brier"], res["desynchronized_donor_brier"])

    def test_memory_helps_on_revisited_blocks(self) -> None:
        res = run_experiment_2_unfamiliar_regimes()
        self.assertLess(
            res["revisited_blocks_with_memory_brier"],
            res["revisited_blocks_no_memory_brier"],
        )


if __name__ == "__main__":
    print("=== Experiment 1: Survival Against Tuned Adaptive Baselines ===")
    for k, v in run_experiment_1_tuned_baselines().items():
        print(f"  {k:<32}: {v:.6f}")

    print("\n=== Experiment 2: Familiar Revisited vs. Unfamiliar Regimes ===")
    for k, v in run_experiment_2_unfamiliar_regimes().items():
        print(f"  {k:<36}: {v:.6f}")

    print("\n=== Experiment 3: Synchronous vs. Desynchronized Donor Controls ===")
    for k, v in run_experiment_3_desynchronized_donor_control().items():
        print(f"  {k:<32}: {v:.6f}")

    print("\nRunning deterministic verification tests...")
    unittest.main(verbosity=2)

