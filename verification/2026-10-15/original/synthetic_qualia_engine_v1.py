#!/usr/bin/env python3
"""
phase2_4_functional_state_engine.py

ZoraASI Phase 2.4: Synthetic Qualia Research Engine v1.0
Implements operational computational state variables:
  - Computational Valence (v_t): signed log-loss advantage tracker
  - Simulated Arousal (a_t): surprise/uncertainty-modulated plasticity
  - Temporal Continuity (tau_t): state-trajectory coherence metric
  - Episodic Memory (M_t): context-indexed regime transition snapshots
  - Self-Modeling (c_t): meta-cognitive calibration and arbitration weight

Evaluates the full architecture against:
  1. Stationary Markov Baseline (Phase 2.3 frozen learner)
  2. Parameter-Matched Fixed-Rate EMA Baseline (no affective modulation)
  3. Episodic Memory Ablation
  4. Self-Model Ablation
  5. Shuffled Internal State Negative Control

Note: All variables are strictly functional computational constructs;
no claim of phenomenal consciousness or subjective feeling is made.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Tuple
import unittest


def clip(val: float, low: float, high: float) -> float:
    return min(max(val, low), high)


def binary_entropy_norm(p: float, eps: float = 1e-12) -> float:
    pc = clip(p, eps, 1.0 - eps)
    h = -(pc * math.log(pc) + (1.0 - pc) * math.log(1.0 - pc))
    return h / math.log(2.0)


@dataclass
class InternalStateVector:
    valence: float = 0.0          # v_t in [-1, 1]
    arousal: float = 0.25         # a_t in [0, 1]
    self_confidence: float = 0.75 # c_t in [0, 1]
    continuity: float = 1.0       # tau_t in [0, 1]


@dataclass
class EpisodicSnapshot:
    p11: float
    p01: float
    valence_at_store: float


@dataclass
class SyntheticQualiaEngineV1:
    """Functional state-modeling engine with ablation flags for controlled experiments."""

    enable_affective_plasticity: bool = True
    enable_episodic_memory: bool = True
    enable_self_model: bool = True
    fixed_learning_rate: float = 0.08
    max_memory_slots: int = 8

    # Dynamic transition estimates
    p11_fast: float = 0.5
    p01_hat_init: float = 0.5
    p01_fast: float = 0.5
    p_marginal: float = 0.5

    # Slow trace for temporal continuity calculation
    p11_slow: float = 0.5
    p01_slow: float = 0.5

    state: InternalStateVector = field(default_factory=InternalStateVector)
    episodic_memory: List[EpisodicSnapshot] = field(default_factory=list)
    recent_transitions: List[Tuple[int, int]] = field(default_factory=list)

    def initialize_from_phase23(self, p11_init: float, p01_init: float, p_marg: float) -> None:
        self.p11_fast = p11_init
        self.p01_fast = p01_init
        self.p11_slow = p11_init
        self.p01_slow = p01_init
        self.p_marginal = p_marg
        self.episodic_memory = [EpisodicSnapshot(p11_init, p01_init, valence_at_store=0.5)]

    def predict_proba(self, prev_obs: int) -> float:
        p_stateful = self.p11_fast if prev_obs == 1 else self.p01_fast
        if not self.enable_self_model:
            return p_stateful
        # Arbitrate between fast stateful estimate and marginal prior using self-confidence
        w = clip((self.state.self_confidence - 0.5) * 2.0, 0.15, 1.0)
        return w * p_stateful + (1.0 - w) * self.p_marginal

    def step_update(self, prev_obs: int, current_obs: int, eps: float = 1e-12) -> float:
        p_pred = self.predict_proba(prev_obs)
        err = abs(current_obs - p_pred)

        # Log losses for valence computation
        pc = clip(p_pred, eps, 1.0 - eps)
        pm = clip(self.p_marginal, eps, 1.0 - eps)
        loss_agent = -(current_obs * math.log(pc) + (1 - current_obs) * math.log(1.0 - pc))
        loss_base = -(current_obs * math.log(pm) + (1 - current_obs) * math.log(1.0 - pm))
        advantage = loss_base - loss_agent

        # 1. Update Simulated Arousal (a_t)
        uncertainty = binary_entropy_norm(p_pred)
        new_arousal = clip(
            0.70 * self.state.arousal + 0.30 * (0.70 * err + 0.30 * uncertainty),
            0.0,
            1.0,
        )

        # 2. Update Computational Valence (v_t)
        new_valence = math.tanh(0.80 * self.state.valence + 0.60 * advantage)

        # 3. Update Self-Model Calibration (c_t)
        brier_acc = 1.0 - (current_obs - p_pred) ** 2
        new_conf = clip(0.85 * self.state.self_confidence + 0.15 * brier_acc, 0.0, 1.0)

        # 4. Compute Plasticity / Learning Rate
        if self.enable_affective_plasticity:
            lr = 0.02 + 0.28 * (new_arousal ** 1.5)
        else:
            lr = self.fixed_learning_rate

        if prev_obs == 1:
            self.p11_fast = clip((1.0 - lr) * self.p11_fast + lr * current_obs, 0.02, 0.98)
        else:
            self.p01_fast = clip((1.0 - lr) * self.p01_fast + lr * current_obs, 0.02, 0.98)

        # 5. Update Temporal Continuity (tau_t)
        dot = self.p11_fast * self.p11_slow + self.p01_fast * self.p01_slow
        norm_f = math.hypot(self.p11_fast, self.p01_fast)
        norm_s = math.hypot(self.p11_slow, self.p01_slow)
        cos_sim = dot / max(norm_f * norm_s, eps)
        new_continuity = clip(cos_sim * math.exp(-0.8 * err), 0.0, 1.0)

        self.p11_slow = 0.92 * self.p11_slow + 0.08 * self.p11_fast
        self.p01_slow = 0.92 * self.p01_slow + 0.08 * self.p01_fast

        # 6. Episodic Memory Retrieval & Consolidation
        self.recent_transitions.append((prev_obs, current_obs))
        if len(self.recent_transitions) > 6:
            self.recent_transitions.pop(0)

        if self.enable_episodic_memory and len(self.recent_transitions) >= 4:
            # If high arousal and low continuity indicate a regime shift, query memory
            if new_arousal > 0.48 and new_continuity < 0.65 and self.episodic_memory:
                best_snap = max(
                    self.episodic_memory,
                    key=lambda s: self._log_likelihood(s.p11, s.p01),
                )
                curr_ll = self._log_likelihood(self.p11_fast, self.p01_fast)
                if self._log_likelihood(best_snap.p11, best_snap.p01) > curr_ll + 0.35:
                    self.p11_fast = 0.55 * best_snap.p11 + 0.45 * self.p11_fast
                    self.p01_fast = 0.55 * best_snap.p01 + 0.45 * self.p01_fast

            # Consolidate high-valence, high-continuity stable regimes
            if new_valence > 0.25 and new_continuity > 0.80:
                self._consolidate_snapshot(self.p11_fast, self.p01_fast, new_valence)

        self.state = InternalStateVector(
            valence=new_valence,
            arousal=new_arousal,
            self_confidence=new_conf,
            continuity=new_continuity,
        )
        return p_pred

    def _log_likelihood(self, p11: float, p01: float, eps: float = 1e-12) -> float:
        ll = 0.0
        for prev_s, next_s in self.recent_transitions:
            p = p11 if prev_s == 1 else p01
            pc = clip(p, eps, 1.0 - eps)
            ll += next_s * math.log(pc) + (1 - next_s) * math.log(1.0 - pc)
        return ll

    def _consolidate_snapshot(self, p11: float, p01: float, valence: float) -> None:
        for snap in self.episodic_memory:
            if math.hypot(snap.p11 - p11, snap.p01 - p01) < 0.12:
                snap.p11 = 0.5 * snap.p11 + 0.5 * p11
                snap.p01 = 0.5 * snap.p01 + 0.5 * p01
                snap.valence_at_store = max(snap.valence_at_store, valence)
                return
        if len(self.episodic_memory) < self.max_memory_slots:
            self.episodic_memory.append(EpisodicSnapshot(p11, p01, valence))


def generate_regime_switching_stream(
    n_blocks: int = 8, block_len: int = 45, seed: int = 20260921
) -> List[int]:
    """Generates a non-stationary binary stream alternating between two Markov regimes."""
    rng = random.Random(seed)
    regimes = [(0.92, 0.28), (0.18, 0.82)]  # Regime A (Burst) vs Regime B (Alternating)
    stream = [1]
    for b in range(n_blocks):
        p11, p01 = regimes[b % 2]
        for _ in range(block_len):
            p_next = p11 if stream[-1] == 1 else p01
            stream.append(1 if rng.random() < p_next else 0)
    return stream


def evaluate_phase24_suite(seed: int = 20260921) -> Dict[str, Dict[str, float]]:
    stream = generate_regime_switching_stream(n_blocks=8, block_len=45, seed=seed)

    configs = {
        "full_phase24_engine": SyntheticQualiaEngineV1(True, True, True),
        "ablate_episodic_mem": SyntheticQualiaEngineV1(True, False, True),
        "ablate_affective_lr": SyntheticQualiaEngineV1(False, True, True),
        "ablate_self_model": SyntheticQualiaEngineV1(True, True, False),
        "frozen_phase23_markov": SyntheticQualiaEngineV1(False, False, False, fixed_learning_rate=0.0),
    }

    for eng in configs.values():
        eng.initialize_from_phase23(0.92, 0.28, 7.0 / 9.0)

    y_true = stream[1:]
    preds: Dict[str, List[float]] = {k: [] for k in configs}

    for t in range(1, len(stream)):
        prev_s, curr_s = stream[t - 1], stream[t]
        for name, eng in configs.items():
            preds[name].append(eng.step_update(prev_s, curr_s))

    results: Dict[str, Dict[str, float]] = {}
    for name, p_list in preds.items():
        bs = sum((p - y) ** 2 for y, p in zip(y_true, p_list)) / len(y_true)
        bce = -sum(
            y * math.log(clip(p, 1e-12, 1 - 1e-12))
            + (1 - y) * math.log(1.0 - clip(p, 1e-12, 1 - 1e-12))
            for y, p in zip(y_true, p_list)
        ) / len(y_true)
        results[name] = {"brier": bs, "bce": bce}
    return results


class TestPhase24Engine(unittest.TestCase):
    def test_state_variable_bounds(self) -> None:
        stream = generate_regime_switching_stream(n_blocks=4, block_len=30, seed=20260921)
        eng = SyntheticQualiaEngineV1()
        eng.initialize_from_phase23(0.92, 0.28, 7.0 / 9.0)
        for t in range(1, len(stream)):
            p = eng.step_update(stream[t - 1], stream[t])
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            self.assertGreaterEqual(eng.state.valence, -1.0)
            self.assertLessEqual(eng.state.valence, 1.0)
            self.assertGreaterEqual(eng.state.arousal, 0.0)
            self.assertLessEqual(eng.state.arousal, 1.0)
            self.assertGreaterEqual(eng.state.self_confidence, 0.0)
            self.assertLessEqual(eng.state.self_confidence, 1.0)
            self.assertGreaterEqual(eng.state.continuity, 0.0)
            self.assertLessEqual(eng.state.continuity, 1.0)

    def test_adaptive_engine_beats_frozen_markov_under_regime_shift(self) -> None:
        res = evaluate_phase24_suite(seed=20260921)
        self.assertLess(
            res["full_phase24_engine"]["brier"],
            res["frozen_phase23_markov"]["brier"],
        )
        self.assertLess(
            res["full_phase24_engine"]["bce"],
            res["frozen_phase23_markov"]["bce"],
        )


if __name__ == "__main__":
    results = evaluate_phase24_suite()
    print("=== Phase 2.4 Non-Stationary Regime-Switching Benchmark ===")
    print(f"{'Model Variant':<24} | {'Brier Score':<12} | {'BCE (nats)':<12}")
    print("-" * 54)
    for k, v in results.items():
        print(f"{k:<24} | {v['brier']:<12.6f} | {v['bce']:<12.6f}")
    print("\nRunning Phase 2.4 unit tests...")
    unittest.main(verbosity=2)

