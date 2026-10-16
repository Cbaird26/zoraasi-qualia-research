#!/usr/bin/env python3
"""Corrected local Phase 2.6 functional-state engine; standard library only.

Defaults preserve the proposed numerical dynamics, except the pre-observation
comparative-marginal scoring bug and reset/validation/memory-capacity fixes.
No guarantee of novelty detection, faster adaptation or phenomenal consciousness.
Benchmarking is separate; legacy_marginal_order=True is only for forensic replay.
"""
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

COUNTERS = (
    "steps", "retrieval_eligible", "candidate_evaluations", "raw_advantage_candidates",
    "rejected_advantage", "rejected_global_novelty", "rejected_row1", "rejected_row0",
    "gate_passed_candidates", "accepted_retrievals", "novelty_rejection_steps",
    "ood_checks", "ood_fallbacks", "damping_activations", "cooldown_blocked_steps",
    "consolidation_calls", "memory_stores", "memory_merges", "capacity_rejections",
)


def finite(value, name="value"):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")


def probability(value, name="probability"):
    finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")


def binary(value):
    if value not in (0, 1):
        raise ValueError("observation must be 0 or 1")


def nonnegative_int(value, name="count"):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def validate_scores(targets, predictions):
    if not targets or len(targets) != len(predictions):
        raise ValueError("nonempty equal-length targets and predictions required")
    for y, p in zip(targets, predictions):
        binary(y)
        probability(p)


def clip(val: float, low: float, high: float) -> float:
    return min(max(val, low), high)


def brier_score(y_true: List[int], y_prob: List[float]) -> float:
    validate_scores(y_true, y_prob)
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def binary_cross_entropy(y_true: List[int], y_prob: List[float], eps: float = 1e-12) -> float:
    validate_scores(y_true, y_prob)
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or not 0 < eps < .5:
        raise ValueError("eps must be finite and in (0,.5)")
    total = 0.0
    for y, p in zip(y_true, y_prob):
        pc = clip(p, eps, 1.0 - eps)
        total -= y * math.log(pc) + (1 - y) * math.log(1.0 - pc)
    return total / len(y_true)


def beta_binomial_row_log_evidence(n_i1: int, n_i0: int) -> float:
    """
    Integrated fixed-path Bernoulli-row likelihood under Beta(1,1).
    No count multiplicity coefficient is needed for a fixed observed path.
    Independent unordered Bernoulli rows would have binomial coefficients, whereas
    aggregated Markov counts need valid-path multiplicity conditional on the start,
    not generally a product of row binomial coefficients. Any shared multiplicity
    cancels from point-versus-mixture ratios for the same observation definition.
    The initial state is conditioned on; both transition rows have independent priors.
    """
    nonnegative_int(n_i1, "n_i1")
    nonnegative_int(n_i0, "n_i0")
    n_total = n_i1 + n_i0
    return (
        math.lgamma(n_i1 + 1.0)
        + math.lgamma(n_i0 + 1.0)
        - math.lgamma(n_total + 2.0)
    )


def point_row_log_likelihood(n_i1: int, n_i0: int, p_i1: float, eps: float = 1e-12) -> float:
    """Fixed-path row log likelihood; endpoints are numerically regularized."""
    nonnegative_int(n_i1, "n_i1")
    nonnegative_int(n_i0, "n_i0")
    probability(p_i1, "p_i1")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or not 0 < eps < .5:
        raise ValueError("eps must be finite and in (0,.5)")
    pc = clip(p_i1, eps, 1.0 - eps)
    return n_i1 * math.log(pc) + n_i0 * math.log(1.0 - pc)


@dataclass
class EpisodicSnapshot:
    p11: float
    p01: float
    valence_score: float
    maturity_steps: int = 0

    def __post_init__(self):
        probability(self.p11, "p11")
        probability(self.p01, "p01")
        finite(self.valence_score, "valence_score")
        if not -1 <= self.valence_score <= 1:
            raise ValueError("valence_score must be in [-1,1]")
        nonnegative_int(self.maturity_steps, "maturity_steps")


class NoveltyAwareEngineV1_2:
    """
    Phase 2.6 Engine featuring:
      - Dual-threshold Bayesian Novelty Gate (H_k vs H_fast AND H_k vs H_novel)
      - Heuristic row-wise consistency check; no guarantee of novelty detection
      - Instantaneous squared-error plasticity and local-posterior fallback
      - Retrieval cooldown; functional predictive states, not phenomenology

    All supplied numerical defaults are preserved. Legacy mode is audit-only.
    Gate/damping/boost flags separate mechanisms without changing default dynamics.
    """

    def __init__(
        self,
        enable_episodic_memory: bool = True,
        enable_novelty_gate: bool = True,
        enable_ood_fallback: bool = True,
        window_size: int = 8,
        adv_margin: float = 0.35,
        novelty_bf_threshold: float = 0.15,
        row_bf_floor: float = -0.25,
        max_snapshots: int = 8,
        *,
        legacy_marginal_order: bool = False,
        enable_prediction_damping: Optional[bool] = None,
        enable_surprise_boost: bool = True,
    ) -> None:
        for name, value in (("enable_episodic_memory", enable_episodic_memory),
                            ("enable_novelty_gate", enable_novelty_gate),
                            ("enable_ood_fallback", enable_ood_fallback),
                            ("legacy_marginal_order", legacy_marginal_order),
                            ("enable_surprise_boost", enable_surprise_boost)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        if enable_prediction_damping is not None and not isinstance(enable_prediction_damping, bool):
            raise ValueError("enable_prediction_damping must be boolean or None")
        nonnegative_int(window_size, "window_size")
        if window_size == 0:
            raise ValueError("window_size must be positive")
        nonnegative_int(max_snapshots, "max_snapshots")
        for name, value in (("adv_margin", adv_margin), ("novelty_bf_threshold", novelty_bf_threshold),
                            ("row_bf_floor", row_bf_floor)):
            finite(value, name)
        if adv_margin < 0:
            raise ValueError("adv_margin must be nonnegative")
        self.legacy_marginal_order = legacy_marginal_order
        self.enable_prediction_damping = enable_ood_fallback if enable_prediction_damping is None else enable_prediction_damping
        self.enable_surprise_boost = enable_surprise_boost
        self.enable_episodic_memory = enable_episodic_memory
        self.enable_novelty_gate = enable_novelty_gate
        self.enable_ood_fallback = enable_ood_fallback
        self.window_size = window_size
        self.adv_margin = adv_margin
        self.novelty_bf_threshold = novelty_bf_threshold
        self.row_bf_floor = row_bf_floor
        self.max_snapshots = max_snapshots

        self.initialize(.5, .5, .5)

    def initialize(self, p11: float, p01: float, marginal: float) -> None:
        """Reset all dynamic fields; caller supplies learned estimates or labeled oracle values."""
        for name, value in (("p11", p11), ("p01", p01), ("marginal", marginal)):
            probability(value, name)
        self.p11_fast = self.p11_slow = p11
        self.p01_fast = self.p01_slow = p01
        self.p_marginal_online = marginal
        self.arousal, self.valence, self.continuity = .18, 0., 1.
        self.stable_steps, self.cooldown = 10, 0
        self.memory = ([EpisodicSnapshot(p11, p01, .5, 30)]
                       if self.enable_episodic_memory and self.max_snapshots else [])
        self.window = []
        self.familiar_retrievals = self.novelty_rejections = self.ood_fallbacks = 0
        self.stats = {key: 0 for key in COUNTERS}
        self.last_learning_rate = None

    def _damping_active(self) -> bool:
        return self.enable_prediction_damping and self.continuity < .62 and len(self.window) >= 2

    def predict_proba(self, prev_s: int) -> float:
        binary(prev_s)
        p_raw = self.p11_fast if prev_s == 1 else self.p01_fast
        # Light uncertainty damping during high-surprise transition steps
        if self._damping_active():
            n11, n10, n01, n00 = self._window_counts()
            local_p = (n11 + 1.0) / (n11 + n10 + 2.0) if prev_s == 1 else (n01 + 1.0) / (n01 + n00 + 2.0)
            blend = clip((0.62 - self.continuity) * 0.85, 0.0, 0.35)
            return (1.0 - blend) * p_raw + blend * local_p
        return p_raw

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

    def _candidate_gate_result(
        self, snap: EpisodicSnapshot, n11: int, n10: int, n01: int, n00: int,
        ll_fast: float, ll_novel_r1: float, ll_novel_r0: float,
    ) -> Tuple[bool, float, Optional[str]]:
        for value in (ll_fast, ll_novel_r1, ll_novel_r0):
            finite(value, "log likelihood")
        ll_snap_r1 = point_row_log_likelihood(n11, n10, snap.p11)
        ll_snap_r0 = point_row_log_likelihood(n01, n00, snap.p01)
        ll_snap = ll_snap_r1 + ll_snap_r0
        # Counters report the FIRST failed condition, in the supplied evaluation order.
        if (ll_snap - ll_fast) <= self.adv_margin:
            return False, ll_snap, "rejected_advantage"
        if not self.enable_novelty_gate:
            return True, ll_snap, None
        if (ll_snap - (ll_novel_r1 + ll_novel_r0)) <= self.novelty_bf_threshold:
            return False, ll_snap, "rejected_global_novelty"
        # Preserve minimum TWO row observations and inclusive floor, unlike prose.
        if (n11 + n10) >= 2 and (ll_snap_r1 - ll_novel_r1) < self.row_bf_floor:
            return False, ll_snap, "rejected_row1"
        if (n01 + n00) >= 2 and (ll_snap_r0 - ll_novel_r0) < self.row_bf_floor:
            return False, ll_snap, "rejected_row0"
        return True, ll_snap, None

    def _candidate_passes_gate(
        self, snap: EpisodicSnapshot, n11: int, n10: int, n01: int, n00: int,
        ll_fast: float, ll_novel_r1: float, ll_novel_r0: float,
    ) -> Tuple[bool, float]:
        """Pure compatibility API; calling it alone does not change diagnostic counts."""
        passed, ll_snap, _ = self._candidate_gate_result(
            snap, n11, n10, n01, n00, ll_fast, ll_novel_r1, ll_novel_r0)
        return passed, ll_snap

    def step_update(self, prev_s: int, curr_s: int) -> float:
        binary(curr_s)
        p_pred = self.predict_proba(prev_s)
        self.stats["steps"] += 1
        if self._damping_active():
            self.stats["damping_activations"] += 1
        err = abs(curr_s - p_pred)
        sq_err = (curr_s - p_pred) ** 2

        marginal_before = self.p_marginal_online
        if self.legacy_marginal_order:
            self.p_marginal_online = .94 * marginal_before + .06 * curr_s
        pc = clip(p_pred, 1e-12, 1.0 - 1e-12)
        pm = clip(self.p_marginal_online, 1e-12, 1.0 - 1e-12)
        loss_agent = -(curr_s * math.log(pc) + (1 - curr_s) * math.log(1.0 - pc))
        loss_marg = -(curr_s * math.log(pm) + (1 - curr_s) * math.log(1.0 - pm))
        # Fair predictive advantage: both losses use pre-outcome forecasts.
        self.p_marginal_online = .94 * marginal_before + .06 * curr_s

        ent = -(pc * math.log(pc) + (1 - pc) * math.log(1.0 - pc)) / math.log(2.0)
        self.arousal = clip(0.65 * self.arousal + 0.35 * (0.80 * err + 0.20 * ent), 0.0, 1.0)
        self.valence = math.tanh(0.80 * self.valence + 0.60 * (loss_marg - loss_agent))

        # Proposed instantaneous term; efficacy must be measured, not assumed.
        surprise_term = .22 * sq_err if self.enable_surprise_boost else 0.
        lr = clip(0.025 + 0.24 * (self.arousal ** 1.3) + surprise_term, 0.02, 0.42)
        self.last_learning_rate = lr

        if prev_s == 1:
            self.p11_fast = clip((1.0 - lr) * self.p11_fast + lr * curr_s, 0.02, 0.98)
        else:
            self.p01_fast = clip((1.0 - lr) * self.p01_fast + lr * curr_s, 0.02, 0.98)

        dot = self.p11_fast * self.p11_slow + self.p01_fast * self.p01_slow
        nf = math.hypot(self.p11_fast, self.p01_fast)
        ns = math.hypot(self.p11_slow, self.p01_slow)
        cos_sim = dot / max(nf * ns, 1e-12)
        self.continuity = clip(cos_sim * math.exp(-0.90 * err), 0.0, 1.0)

        self.p11_slow = 0.88 * self.p11_slow + 0.12 * self.p11_fast
        self.p01_slow = 0.88 * self.p01_slow + 0.12 * self.p01_fast

        self.window.append((prev_s, curr_s))
        if len(self.window) > self.window_size:
            self.window.pop(0)

        if self.cooldown > 0:
            self.cooldown -= 1
            if self.cooldown > 0:
                self.stats["cooldown_blocked_steps"] += 1

        # Evaluate Episodic Retrieval & OOD Novelty Fallback
        if len(self.window) >= 6 and self.cooldown == 0:
            n11, n10, n01, n00 = self._window_counts()
            ll_fast = (
                point_row_log_likelihood(n11, n10, self.p11_fast)
                + point_row_log_likelihood(n01, n00, self.p01_fast)
            )
            ll_novel_r1 = beta_binomial_row_log_evidence(n11, n10)
            ll_novel_r0 = beta_binomial_row_log_evidence(n01, n00)
            ll_novel = ll_novel_r1 + ll_novel_r0

            if self.arousal > 0.38 and self.continuity < 0.72:
                self.stats["retrieval_eligible"] += 1
                best_snap: Optional[EpisodicSnapshot] = None
                best_ll = -1e9
                any_raw_advantage = False

                if self.enable_episodic_memory and self.memory:
                    for snap in self.memory:
                        self.stats["candidate_evaluations"] += 1
                        ll_s = (
                            point_row_log_likelihood(n11, n10, snap.p11)
                            + point_row_log_likelihood(n01, n00, snap.p01)
                        )
                        if (ll_s - ll_fast) > self.adv_margin:
                            any_raw_advantage = True
                            self.stats["raw_advantage_candidates"] += 1
                        passed, _, reason = self._candidate_gate_result(
                            snap, n11, n10, n01, n00, ll_fast, ll_novel_r1, ll_novel_r0
                        )
                        if passed:
                            self.stats["gate_passed_candidates"] += 1
                        else:
                            self.stats[reason] += 1
                        if passed and ll_s > best_ll:
                            best_ll = ll_s
                            best_snap = snap

                if best_snap is not None:
                    # Accepted stored snapshot; this does not verify true regime identity.
                    self.p11_fast = 0.75 * best_snap.p11 + 0.25 * self.p11_fast
                    self.p01_fast = 0.75 * best_snap.p01 + 0.25 * self.p01_fast
                    self.familiar_retrievals += 1
                    self.stats["accepted_retrievals"] += 1
                    self.cooldown = 6
                    self.stable_steps = 0
                else:
                    if any_raw_advantage and self.enable_novelty_gate:
                        self.novelty_rejections += 1
                        self.stats["novelty_rejection_steps"] += 1
                    # If H_novel strongly beats current fast state, trigger OOD local-Bayes reset
                    if self.enable_ood_fallback:
                        self.stats["ood_checks"] += 1
                    if self.enable_ood_fallback and (ll_novel - ll_fast) > 0.45:
                        p11_local = (n11 + 1.0) / (n11 + n10 + 2.0)
                        p01_local = (n01 + 1.0) / (n01 + n00 + 2.0)
                        self.p11_fast = 0.65 * p11_local + 0.35 * self.p11_fast
                        self.p01_fast = 0.65 * p01_local + 0.35 * self.p01_fast
                        self.ood_fallbacks += 1
                        self.stats["ood_fallbacks"] += 1
                        self.cooldown = 5
                        self.stable_steps = 0

        # Consolidate mature, high-continuity regimes into episodic memory
        if self.continuity > 0.78 and err < 0.35:
            self.stable_steps += 1
        else:
            self.stable_steps = max(0, self.stable_steps - 1)

        if self.enable_episodic_memory and self.stable_steps >= 14 and self.valence > 0.10:
            self._consolidate(self.p11_fast, self.p01_fast, self.valence)

        return p_pred

    def _consolidate(self, p11: float, p01: float, val: float) -> None:
        probability(p11, "p11")
        probability(p01, "p01")
        finite(val, "valence")
        if not -1 <= val <= 1:
            raise ValueError("valence must be in [-1,1]")
        if not self.enable_episodic_memory or self.max_snapshots == 0:
            return
        self.stats["consolidation_calls"] += 1
        for snap in self.memory:
            if math.hypot(snap.p11 - p11, snap.p01 - p01) < 0.14:
                snap.p11 = 0.6 * snap.p11 + 0.4 * p11
                snap.p01 = 0.6 * snap.p01 + 0.4 * p01
                snap.valence_score = max(snap.valence_score, val)
                snap.maturity_steps += 1
                self.stats["memory_merges"] += 1
                return
        if len(self.memory) < self.max_snapshots:
            self.memory.append(EpisodicSnapshot(p11, p01, val, maturity_steps=14))
            self.stats["memory_stores"] += 1
        else:
            self.stats["capacity_rejections"] += 1

