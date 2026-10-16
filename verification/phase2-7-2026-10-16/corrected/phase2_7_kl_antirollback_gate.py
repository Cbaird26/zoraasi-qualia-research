#!/usr/bin/env python3
"""Corrected local Phase 2.7 engine; original artifacts remain separate.

Functional prediction experiment, not phenomenal or physical-theory evidence.
The supplied proposal changes storage as well as retrieval. Explicit phase25
storage/sequential-likelihood mode permits exact pinned-learner comparisons.
"""
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

COUNTERS = ("steps", "retrieval_eligible", "candidate_evaluations", "raw_advantage_candidates",
            "rejected_fast", "rejected_slow", "rejected_distance", "rejected_missing_rows",
            "rejected_kl_row1", "rejected_kl_row0", "gate_passed_candidates", "accepted_retrievals",
            "raw_retrievals_rejected", "consolidation_calls", "memory_stores", "memory_merges", "capacity_rejections")


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


def epsilon(value):
    finite(value, "eps")
    if not 0 < value < .5:
        raise ValueError("eps must be in (0,.5)")


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
    epsilon(eps)
    total = 0.0
    for y, p in zip(y_true, y_prob):
        pc = clip(p, eps, 1.0 - eps)
        total -= y * math.log(pc) + (1 - y) * math.log(1.0 - pc)
    return total / len(y_true)


def bernoulli_kl(p: float, q: float, eps: float = 1e-12) -> float:
    """Bernoulli KL in nats; endpoint probabilities are numerically regularized."""
    probability(p, "p")
    probability(q, "q")
    epsilon(eps)
    pc = clip(p, eps, 1.0 - eps)
    qc = clip(q, eps, 1.0 - eps)
    return pc * math.log(pc / qc) + (1.0 - pc) * math.log((1.0 - pc) / (1.0 - qc))


def jeffreys_row_kl(n_i1: int, n_i0: int, p_snap_i: float) -> float:
    """
    Computes D_KL(p_tilde_i || p_snap_i) where p_tilde_i = (n_i1 + 0.5) / (n_i1 + n_i0 + 1.0)
    is the regularizing Jeffreys Beta(0.5, 0.5) posterior mean on row i.
    """
    nonnegative_int(n_i1, "n_i1")
    nonnegative_int(n_i0, "n_i0")
    probability(p_snap_i, "p_snap_i")
    n_tot = n_i1 + n_i0
    p_tilde = (n_i1 + 0.5) / (n_tot + 1.0)
    return bernoulli_kl(p_tilde, p_snap_i)


def point_row_log_likelihood(n_i1: int, n_i0: int, p_i1: float, eps: float = 1e-12) -> float:
    nonnegative_int(n_i1, "n_i1")
    nonnegative_int(n_i0, "n_i0")
    probability(p_i1, "p_i1")
    epsilon(eps)
    pc = clip(p_i1, eps, 1.0 - eps)
    return n_i1 * math.log(pc) + n_i0 * math.log(1.0 - pc)


@dataclass
class EpisodicSnapshot:
    p11: float
    p01: float
    valence_score: float

    def __post_init__(self):
        probability(self.p11, "p11")
        probability(self.p01, "p01")
        finite(self.valence_score, "valence_score")
        if not -1 <= self.valence_score <= 1:
            raise ValueError("valence_score must be in [-1,1]")


@dataclass
class ScalarTrace:
    arousal: List[float] = field(default_factory=list)
    valence: List[float] = field(default_factory=list)
    continuity: List[float] = field(default_factory=list)


class Phase27Engine:
    """Proposed Phase 2.7 learner with explicit storage and retrieval ablations.

    consolidation_mode="proposal" preserves supplied stability/distance storage rules.
    consolidation_mode="phase25" uses the pinned Phase 2.5 storage/merge rules and
    defaults to its per-transition likelihood arithmetic. With phase25_ungated,
    window_size=6 and margin=.30 it reproduces prior predictive states exactly.
    The gate is a heuristic, not a guarantee of regime identity or consciousness.
    """

    def __init__(
        self,
        mode: str = "phase27_kl_gated",
        window_size: int = 6,
        retrieval_ll_margin: float = 0.30,
        kl_row_threshold: float = 0.12,
        anti_rollback_dist: float = 0.20,
        max_snapshots: int = 8,
        *,
        consolidation_mode: str = "proposal",
        enable_antirollback: bool = True,
        enable_kl_gate: bool = True,
        legacy_marginal_order: bool = False,
        legacy_slow_distance: bool = False,
        likelihood_mode: Optional[str] = None,
    ) -> None:
        if mode not in ("phase27_kl_gated", "phase25_ungated", "no_memory"):
            raise ValueError(f"Unknown engine mode: {mode}")
        if consolidation_mode not in ("proposal", "phase25"):
            raise ValueError("consolidation_mode must be proposal or phase25")
        if likelihood_mode is None:
            likelihood_mode = "sequential" if consolidation_mode == "phase25" else "grouped"
        if likelihood_mode not in ("sequential", "grouped"):
            raise ValueError("likelihood_mode must be sequential or grouped")
        for name, flag in (("enable_antirollback", enable_antirollback), ("enable_kl_gate", enable_kl_gate),
                           ("legacy_marginal_order", legacy_marginal_order), ("legacy_slow_distance", legacy_slow_distance)):
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be boolean")
        nonnegative_int(window_size, "window_size")
        nonnegative_int(max_snapshots, "max_snapshots")
        if window_size == 0:
            raise ValueError("window_size must be positive")
        for name, value in (("retrieval_ll_margin", retrieval_ll_margin), ("kl_row_threshold", kl_row_threshold),
                            ("anti_rollback_dist", anti_rollback_dist)):
            finite(value, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        self.consolidation_mode = consolidation_mode
        self.enable_antirollback = enable_antirollback
        self.enable_kl_gate = enable_kl_gate
        self.legacy_marginal_order = legacy_marginal_order
        self.legacy_slow_distance = legacy_slow_distance
        self.likelihood_mode = likelihood_mode
        self.mode = mode
        self.window_size = window_size
        self.retrieval_ll_margin = retrieval_ll_margin
        self.kl_row_threshold = kl_row_threshold
        self.anti_rollback_dist = anti_rollback_dist
        self.max_snapshots = max_snapshots

        self.initialize(.5, .5, .5)

    def initialize(self, p11: float, p01: float, marginal: float) -> None:
        """Reset the run; caller provides learned estimates or explicitly labeled truth."""
        for name, value in (("p11", p11), ("p01", p01), ("marginal", marginal)):
            probability(value, name)
        self.p11_fast = self.p11_slow = p11
        self.p01_fast = self.p01_slow = p01
        self.p_marginal_online = marginal
        self.arousal, self.valence, self.continuity = .20, 0., 1.
        self.stable_steps = 10
        self.memory = ([EpisodicSnapshot(p11, p01, .5)]
                       if self.mode != "no_memory" and self.max_snapshots else [])
        self.window = []
        self.recorded_trace = ScalarTrace()
        self.retrievals_accepted = self.retrievals_rejected_by_kl_or_rollback = 0
        self.stats = {key: 0 for key in COUNTERS}
        self.last_gate_slow_reference = self.last_gate_likelihood_reference = None
        self.last_gate_ll_slow = None
        self.last_learning_rate = None

    def scalars(self) -> Tuple[float, float, float]:
        return self.arousal, self.valence, self.continuity

    def predict_proba(self, prev_s: int) -> float:
        binary(prev_s)
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
        probability(p11, "p11")
        probability(p01, "p01")
        if self.likelihood_mode == "sequential":
            # Preserve the pinned Phase 2.5 summation order, not just its algebra.
            ll = 0.0
            for prv, nxt in self.window:
                p = clip(p11 if prv == 1 else p01, 1e-12, 1.0 - 1e-12)
                ll += nxt * math.log(p) + (1 - nxt) * math.log(1.0 - p)
            return ll
        n11, n10, n01, n00 = self._window_counts()
        return point_row_log_likelihood(n11, n10, p11) + point_row_log_likelihood(n01, n00, p01)

    def _relative_passes(self, candidate_ll: float, reference_ll: float) -> bool:
        if self.likelihood_mode == "sequential":
            return candidate_ll > reference_ll + self.retrieval_ll_margin
        return (candidate_ll - reference_ll) > self.retrieval_ll_margin

    def _candidate_gate_result(
        self, snap: EpisodicSnapshot, n11: int, n10: int, n01: int, n00: int,
        ll_fast: float, ll_slow: float, *, slow_reference=None, candidate_ll=None,
    ) -> Tuple[bool, Optional[str]]:
        for count in (n11, n10, n01, n00):
            nonnegative_int(count)
        finite(ll_fast, "ll_fast")
        finite(ll_slow, "ll_slow")
        if slow_reference is None:
            slow_reference = (self.p11_slow, self.p01_slow)
        if len(slow_reference) != 2:
            raise ValueError("slow_reference must have two probabilities")
        for value in slow_reference:
            probability(value, "slow_reference")
        ll_snap = (point_row_log_likelihood(n11, n10, snap.p11)
                   + point_row_log_likelihood(n01, n00, snap.p01)) if candidate_ll is None else candidate_ll
        finite(ll_snap, "candidate_ll")
        if self.mode == "no_memory":
            return False, "disabled"
        if not self._relative_passes(ll_snap, ll_fast):
            return False, "rejected_fast"
        if self.mode == "phase25_ungated":
            return True, None
        if self.enable_antirollback:
            if not self._relative_passes(ll_snap, ll_slow):
                return False, "rejected_slow"
            if math.hypot(snap.p11-slow_reference[0], snap.p01-slow_reference[1]) < self.anti_rollback_dist:
                return False, "rejected_distance"
        if self.enable_kl_gate:
            if (n11+n10) < 1 or (n01+n00) < 1:
                return False, "rejected_missing_rows"
            if jeffreys_row_kl(n11, n10, snap.p11) > self.kl_row_threshold:
                return False, "rejected_kl_row1"
            if jeffreys_row_kl(n01, n00, snap.p01) > self.kl_row_threshold:
                return False, "rejected_kl_row0"
        return True, None

    def candidate_passes_phase27_gate(
        self, snap: EpisodicSnapshot, n11: int, n10: int, n01: int, n00: int,
        ll_fast: float, ll_slow: float, *, slow_reference=None,
    ) -> bool:
        """Pure count-form gate API; optional slow tuple makes reference timing explicit.

        For step updates in sequential compatibility mode, ordered candidate likelihoods
        are supplied internally. Direct count-only calls use grouped count likelihoods.
        """
        return self._candidate_gate_result(snap, n11, n10, n01, n00, ll_fast, ll_slow,
                                           slow_reference=slow_reference)[0]

    def step_update(self, prev_s: int, curr_s: int) -> float:
        binary(curr_s)
        p_pred = self.predict_proba(prev_s)
        err = abs(curr_s - p_pred)
        self.stats["steps"] += 1

        # Compare both predictors before either sees the current outcome.
        marginal_before = self.p_marginal_online
        if self.legacy_marginal_order:
            self.p_marginal_online = .95 * marginal_before + .05 * curr_s
        pc = clip(p_pred, 1e-12, 1.0 - 1e-12)
        pm = clip(self.p_marginal_online, 1e-12, 1.0 - 1e-12)
        loss_agent = -(curr_s * math.log(pc) + (1 - curr_s) * math.log(1.0 - pc))
        loss_marg = -(curr_s * math.log(pm) + (1 - curr_s) * math.log(1.0 - pm))
        self.p_marginal_online = .95 * marginal_before + .05 * curr_s

        ent = -(pc * math.log(pc) + (1 - pc) * math.log(1.0 - pc)) / math.log(2.0)
        endo_arousal = clip(0.70 * self.arousal + 0.30 * (0.75 * err + 0.25 * ent), 0.0, 1.0)
        endo_valence = math.tanh(0.80 * self.valence + 0.60 * (loss_marg - loss_agent))

        dot = self.p11_fast * self.p11_slow + self.p01_fast * self.p01_slow
        nf = math.hypot(self.p11_fast, self.p01_fast)
        ns = math.hypot(self.p11_slow, self.p01_slow)
        cos_sim = dot / max(nf * ns, 1e-12)
        endo_continuity = clip(cos_sim * math.exp(-0.85 * err), 0.0, 1.0)

        self.recorded_trace.arousal.append(endo_arousal)
        self.recorded_trace.valence.append(endo_valence)
        self.recorded_trace.continuity.append(endo_continuity)

        # Preserve the proposed Phase 2.5-form learning rate.
        lr = 0.02 + 0.32 * (endo_arousal ** 1.4)
        self.last_learning_rate = lr
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

        # No-memory disables both retrieval and storage regardless of component flags.
        if self.mode != "no_memory" and len(self.window) == self.window_size:
            if endo_arousal > 0.44 and endo_continuity < 0.68 and self.memory:
                self.stats["retrieval_eligible"] += 1
                n11, n10, n01, n00 = self._window_counts()
                ll_fast = self._window_ll(self.p11_fast, self.p01_fast)
                slow_before = (prev_p11_slow, prev_p01_slow)
                ll_slow = self._window_ll(*slow_before)
                distance_reference = ((self.p11_slow, self.p01_slow)
                                      if self.legacy_slow_distance else slow_before)
                self.last_gate_slow_reference = distance_reference
                self.last_gate_likelihood_reference = slow_before
                self.last_gate_ll_slow = ll_slow
                likelihoods = [self._window_ll(s.p11, s.p01) for s in self.memory]
                raw_index = max(range(len(self.memory)), key=likelihoods.__getitem__)
                # Keep supplied diagnostic comparison arithmetic for forensic replay.
                raw_would_fire = likelihoods[raw_index] > ll_fast + self.retrieval_ll_margin
                valid_indices = []
                for i, snap in enumerate(self.memory):
                    self.stats["candidate_evaluations"] += 1
                    if self._relative_passes(likelihoods[i], ll_fast):
                        self.stats["raw_advantage_candidates"] += 1
                    passed, reason = self._candidate_gate_result(
                        snap, n11, n10, n01, n00, ll_fast, ll_slow,
                        slow_reference=distance_reference, candidate_ll=likelihoods[i])
                    if passed:
                        self.stats["gate_passed_candidates"] += 1
                        valid_indices.append(i)
                    else:
                        self.stats[reason] += 1
                if valid_indices:
                    best_index = max(valid_indices, key=likelihoods.__getitem__)
                    best_snap = self.memory[best_index]
                    self.p11_fast = 0.65 * best_snap.p11 + 0.35 * self.p11_fast
                    self.p01_fast = 0.65 * best_snap.p01 + 0.35 * self.p01_fast
                    self.retrievals_accepted += 1
                    self.stats["accepted_retrievals"] += 1
                elif raw_would_fire:
                    self.retrievals_rejected_by_kl_or_rollback += 1
                    self.stats["raw_retrievals_rejected"] += 1

            store_allowed = endo_valence > .20 and endo_continuity > .82
            if self.consolidation_mode == "proposal":
                store_allowed = (store_allowed and self.stable_steps >= 10 and
                    math.hypot(self.p11_fast-self.p11_slow, self.p01_fast-self.p01_slow) < .08)
            if store_allowed:
                self._consolidate(self.p11_fast, self.p01_fast, endo_valence)

        self.arousal = endo_arousal
        self.valence = endo_valence
        self.continuity = endo_continuity
        return p_pred

    def _consolidate(self, p11: float, p01: float, val: float) -> None:
        probability(p11, "p11")
        probability(p01, "p01")
        finite(val, "valence")
        if not -1 <= val <= 1:
            raise ValueError("valence must be in [-1,1]")
        if self.mode == "no_memory" or self.max_snapshots == 0:
            return
        self.stats["consolidation_calls"] += 1
        radius = .12 if self.consolidation_mode == "phase25" else .16
        for snap in self.memory:
            if math.hypot(snap.p11 - p11, snap.p01 - p01) < radius:
                snap.p11 = 0.5 * snap.p11 + 0.5 * p11
                snap.p01 = 0.5 * snap.p01 + 0.5 * p01
                snap.valence_score = max(snap.valence_score, val)
                self.stats["memory_merges"] += 1
                return
        if len(self.memory) < self.max_snapshots:
            self.memory.append(EpisodicSnapshot(p11, p01, val))
            self.stats["memory_stores"] += 1
        else:
            self.stats["capacity_rejections"] += 1

