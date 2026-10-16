#!/usr/bin/env python3
"""Corrected local Phase 2.8 functional prediction engine; standard library only.

The supplied surprise, row-window, KL, direction and cooldown rules are heuristics.
They are not a calibrated epistemic oracle or a guarantee of regime-clean memory.
Original artifacts remain separate. No claim about subjective experience follows.
"""
from dataclasses import dataclass, field
import math
from typing import List, Tuple

COUNTERS = (
    "steps", "memory_disabled_steps", "missing_memory_steps", "window_blocked_steps",
    "insufficient_row_steps", "cooldown_blocked_steps", "surprise_blocked_steps",
    "arousal_blocked_steps", "continuity_blocked_steps", "retrieval_eligible",
    "candidate_evaluations", "raw_advantage_candidates", "rejected_fast",
    "rejected_slow", "rejected_kl", "rejected_directional", "gate_passed_candidates",
    "accepted_retrievals", "rejected_retrieval_steps", "consolidation_calls",
    "memory_stores", "memory_merges", "capacity_rejections",
)


def finite(value, name="value"):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")


def probability(value, name="probability"):
    finite(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")


def binary(value):
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("observation must be an integer 0 or 1")


def nonnegative_int(value, name="count"):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def epsilon(value):
    finite(value, "eps")
    if not 0 < value < .5:
        raise ValueError("eps must be in (0,.5)")


def clip(val, low, high):
    return min(max(val, low), high)


def validate_scores(y_true, y_prob):
    if not y_true or len(y_true) != len(y_prob):
        raise ValueError("nonempty equal-length targets and predictions required")
    for y, p in zip(y_true, y_prob):
        binary(y)
        probability(p)


def brier_score(y_true, y_prob):
    validate_scores(y_true, y_prob)
    return sum((p-y)**2 for y, p in zip(y_true, y_prob))/len(y_true)


def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    validate_scores(y_true, y_prob)
    epsilon(eps)
    total = 0.
    for y, p in zip(y_true, y_prob):
        pc = clip(p, eps, 1.-eps)
        total -= y*math.log(pc)+(1-y)*math.log(1.-pc)
    return total/len(y_true)


def bernoulli_kl(p, q, eps=1e-12):
    """Bernoulli KL in nats, with finite numerical regularization at endpoints."""
    probability(p, "p")
    probability(q, "q")
    epsilon(eps)
    pc, qc = clip(p, eps, 1.-eps), clip(q, eps, 1.-eps)
    return pc*math.log(pc/qc)+(1.-pc)*math.log((1.-pc)/(1.-qc))


def jeffreys_row_kl(n_i1, n_i0, p_snap_i):
    nonnegative_int(n_i1, "n_i1")
    nonnegative_int(n_i0, "n_i0")
    return bernoulli_kl((n_i1+.5)/(n_i1+n_i0+1.), p_snap_i)


def point_row_ll(n_i1, n_i0, p_i1, eps=1e-12):
    nonnegative_int(n_i1, "n_i1")
    nonnegative_int(n_i0, "n_i0")
    probability(p_i1, "p_i1")
    epsilon(eps)
    pc = clip(p_i1, eps, 1.-eps)
    return n_i1*math.log(pc)+n_i0*math.log(1.-pc)


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


class Phase28Engine:
    """Pinned Phase 2.5 learning/storage with explicit row retrieval ablations.

    phase25_ungated reproduces pinned Phase 2.5 forecasts and predictive state.
    no_memory disables retrieval/storage. Constructor initialization is neutral;
    initialize(...) accepts independently learned estimates and fully resets state.
    legacy_marginal_order reproduces the supplied in-sample marginal scoring bug.
    All ablation flags act only in phase28_epistemic_row retrieval, not storage.
    """
    def __init__(
        self, mode="phase28_epistemic_row", retrieval_ll_margin=.30,
        row_kl_threshold=.065, min_row_obs=4, epistemic_surprise_thresh=.08,
        row_cooldown_steps=12, max_snapshots=8, *, enable_surprise=True,
        enable_kl=True, enable_directional=True, enable_cooldown=True,
        enable_slow_advantage=True, apply_row_only=True, legacy_marginal_order=False,
    ):
        if mode not in ("phase28_epistemic_row", "phase25_ungated", "no_memory"):
            raise ValueError(f"Unknown mode: {mode}")
        for name, value in (("retrieval_ll_margin", retrieval_ll_margin),
                            ("row_kl_threshold", row_kl_threshold),
                            ("epistemic_surprise_thresh", epistemic_surprise_thresh)):
            finite(value, name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name, value in (("min_row_obs", min_row_obs),
                            ("row_cooldown_steps", row_cooldown_steps),
                            ("max_snapshots", max_snapshots)):
            nonnegative_int(value, name)
        if min_row_obs == 0:
            raise ValueError("min_row_obs must be positive")
        for name, flag in (("enable_surprise", enable_surprise), ("enable_kl", enable_kl),
                           ("enable_directional", enable_directional),
                           ("enable_cooldown", enable_cooldown),
                           ("enable_slow_advantage", enable_slow_advantage),
                           ("apply_row_only", apply_row_only),
                           ("legacy_marginal_order", legacy_marginal_order)):
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, flag)
        self.mode, self.retrieval_ll_margin = mode, retrieval_ll_margin
        self.row_kl_threshold, self.min_row_obs = row_kl_threshold, min_row_obs
        self.epistemic_surprise_thresh = epistemic_surprise_thresh
        self.row_cooldown_steps, self.max_snapshots = row_cooldown_steps, max_snapshots
        self.initialize(.5, .5, .5)

    def initialize(self, p11, p01, marginal):
        for value in (p11, p01, marginal):
            probability(value)
        self.p11_fast = self.p11_slow = p11
        self.p01_fast = self.p01_slow = p01
        self.p_marginal_online = marginal
        self.arousal, self.valence, self.continuity = .20, 0., 1.
        self.epistemic_surprise = 0.
        self.row_cooldown = [0, 0]
        self.memory = ([EpisodicSnapshot(p11, p01, .5)]
                       if self.mode != "no_memory" and self.max_snapshots else [])
        self.window6 = []
        self.row_buffers = [[], []]
        self.row_buffer_indices = [[], []]
        self.row_retrievals = self.rejected_candidates = 0
        self.recorded_trace = ScalarTrace()
        self.stats = dict.fromkeys(COUNTERS, 0)
        self.last_step = None

    def predict_proba(self, prev_s):
        binary(prev_s)
        return self.p11_fast if prev_s == 1 else self.p01_fast

    def _window6_ll(self, p11, p01):
        ll = 0.
        for prev, y in self.window6:
            p = clip(p11 if prev == 1 else p01, 1e-12, 1.-1e-12)
            ll += y*math.log(p)+(1-y)*math.log(1.-p)
        return ll

    @staticmethod
    def passes_directional_contraction(p_cand, p_fast, p_slow, n_i1, n_i0, curr_s):
        """Supplied three-part heuristic; sufficient for contraction to THIS row MLE.

        It is not necessary for contraction, nor does it identify a true regime.
        """
        for p in (p_cand, p_fast, p_slow):
            probability(p)
        nonnegative_int(n_i1, "n_i1")
        nonnegative_int(n_i0, "n_i0")
        binary(curr_s)
        total = n_i1+n_i0
        if total == 0:
            return False
        mle = n_i1/total
        if (p_cand-p_fast)*(curr_s-p_fast) <= 0.:
            return False
        if abs(p_cand-mle) >= abs(p_fast-mle):
            return False
        if abs(p_cand-p_slow) < .18 and abs(mle-p_slow) > .22:
            return False
        return True

    def _row_candidate_reason(self, p_cand, p_fast, p_slow, ones, zeros, curr_s):
        ll = point_row_ll(ones, zeros, p_cand)
        if not (ll-point_row_ll(ones, zeros, p_fast)) > self.retrieval_ll_margin:
            return "fast", ll
        if self.enable_slow_advantage and not (ll-point_row_ll(ones, zeros, p_slow)) > self.retrieval_ll_margin:
            return "slow", ll
        if self.enable_kl and jeffreys_row_kl(ones, zeros, p_cand) > self.row_kl_threshold:
            return "kl", ll
        if self.enable_directional and not self.passes_directional_contraction(p_cand, p_fast, p_slow, ones, zeros, curr_s):
            return "directional", ll
        return None, ll

    def step_update(self, prev_s, curr_s):
        binary(prev_s)
        binary(curr_s)
        # Validated before ANY state mutation; returned forecast precedes curr_s.
        p_pred = self.predict_proba(prev_s)
        step_index = self.stats["steps"]
        stats_before = dict(self.stats)
        cooldown_before = tuple(self.row_cooldown)
        err = abs(curr_s-p_pred)
        surprise_step = (curr_s-p_pred)**2-p_pred*(1.-p_pred)
        self.epistemic_surprise = .78*self.epistemic_surprise+.22*surprise_step
        marginal_before = self.p_marginal_online
        if self.legacy_marginal_order:
            self.p_marginal_online = .95*marginal_before+.05*curr_s
        pc = clip(p_pred, 1e-12, 1.-1e-12)
        pm = clip(self.p_marginal_online, 1e-12, 1.-1e-12)
        loss_agent = -(curr_s*math.log(pc)+(1-curr_s)*math.log(1.-pc))
        loss_marg = -(curr_s*math.log(pm)+(1-curr_s)*math.log(1.-pm))
        self.p_marginal_online = .95*marginal_before+.05*curr_s
        ent = -(pc*math.log(pc)+(1.-pc)*math.log(1.-pc))/math.log(2.)
        arousal = clip(.70*self.arousal+.30*(.75*err+.25*ent), 0., 1.)
        valence = math.tanh(.80*self.valence+.60*(loss_marg-loss_agent))
        dot = self.p11_fast*self.p11_slow+self.p01_fast*self.p01_slow
        nf, ns = math.hypot(self.p11_fast, self.p01_fast), math.hypot(self.p11_slow, self.p01_slow)
        cosine = dot/max(nf*ns, 1e-12)
        continuity = clip(cosine*math.exp(-.85*err), 0., 1.)
        self.recorded_trace.arousal.append(arousal)
        self.recorded_trace.valence.append(valence)
        self.recorded_trace.continuity.append(continuity)
        lr = .02+.32*(arousal**1.4)
        if prev_s == 1:
            self.p11_fast = clip((1.-lr)*self.p11_fast+lr*curr_s, .02, .98)
        else:
            self.p01_fast = clip((1.-lr)*self.p01_fast+lr*curr_s, .02, .98)
        slow_before = (self.p11_slow, self.p01_slow)
        self.p11_slow = .90*self.p11_slow+.10*self.p11_fast
        self.p01_slow = .90*self.p01_slow+.10*self.p01_fast
        self.window6.append((prev_s, curr_s))
        if len(self.window6) > 6:
            self.window6.pop(0)
        self.row_buffers[prev_s].append(curr_s)
        self.row_buffer_indices[prev_s].append(step_index)
        if len(self.row_buffers[prev_s]) > self.min_row_obs:
            self.row_buffers[prev_s].pop(0)
            self.row_buffer_indices[prev_s].pop(0)
        # Both row clocks count GLOBAL transitions, not observations of that row.
        for row in (0, 1):
            if self.row_cooldown[row] > 0:
                self.row_cooldown[row] -= 1
        cooldown_at_gate = tuple(self.row_cooldown)
        pre_retrieval = (self.p11_fast, self.p01_fast)
        surprise_trigger = self.epistemic_surprise > self.epistemic_surprise_thresh
        blocked = None
        if self.mode == "no_memory" or self.max_snapshots == 0:
            blocked = "memory_disabled"
        elif not self.memory:
            blocked = "missing_memory"
        elif self.mode == "phase25_ungated" and len(self.window6) < 6:
            blocked = "window_blocked"
        elif self.mode == "phase28_epistemic_row" and len(self.row_buffers[prev_s]) < self.min_row_obs:
            blocked = "insufficient_row"
        elif self.mode == "phase28_epistemic_row" and self.enable_cooldown and self.row_cooldown[prev_s] > 0:
            blocked = "cooldown_blocked"
        elif self.mode == "phase28_epistemic_row" and self.enable_surprise and not surprise_trigger:
            blocked = "surprise_blocked"
        elif arousal <= .44:
            blocked = "arousal_blocked"
        elif continuity >= .68:
            blocked = "continuity_blocked"
        eligible = blocked is None
        chosen_index = chosen_probs = None
        candidate_records = []
        if blocked:
            self.stats[blocked+"_steps"] += 1
        else:
            self.stats["retrieval_eligible"] += 1
            best_ll, best_index = -1e9, None
            if self.mode == "phase25_ungated":
                fast_ll = self._window6_ll(*pre_retrieval)
                for idx, snap in enumerate(self.memory):
                    ll = self._window6_ll(snap.p11, snap.p01)
                    reason = None if ll > fast_ll+self.retrieval_ll_margin else "fast"
                    self._record_candidate(reason)
                    candidate_records.append({"index": idx, "first_rejection": reason, "ll": ll})
                    if reason is None and ll > best_ll:
                        best_ll, best_index = ll, idx
            else:
                ones = sum(self.row_buffers[prev_s])
                zeros = len(self.row_buffers[prev_s])-ones
                fast_i = pre_retrieval[0 if prev_s else 1]
                slow_i = slow_before[0 if prev_s else 1]
                for idx, snap in enumerate(self.memory):
                    cand = snap.p11 if prev_s else snap.p01
                    reason, ll = self._row_candidate_reason(cand, fast_i, slow_i, ones, zeros, curr_s)
                    self._record_candidate(reason)
                    candidate_records.append({"index": idx, "first_rejection": reason, "ll": ll})
                    if reason is None and ll > best_ll:
                        best_ll, best_index = ll, idx
            if best_index is not None:
                chosen_index = best_index
                snap = self.memory[chosen_index]
                chosen_probs = (snap.p11, snap.p01)  # capture BEFORE consolidation/merge
                if self.mode == "phase25_ungated" or not self.apply_row_only or prev_s == 1:
                    self.p11_fast = .65*snap.p11+.35*self.p11_fast
                if self.mode == "phase25_ungated" or not self.apply_row_only or prev_s == 0:
                    self.p01_fast = .65*snap.p01+.35*self.p01_fast
                if self.mode == "phase28_epistemic_row":
                    self.row_cooldown[prev_s] = self.row_cooldown_steps
                self.row_retrievals += 1
                self.stats["accepted_retrievals"] += 1
            else:
                self.stats["rejected_retrieval_steps"] += 1
                if self.mode == "phase28_epistemic_row":
                    self.rejected_candidates += 1  # source counter is per eligible STEP
        after_retrieval = (self.p11_fast, self.p01_fast)
        if self.mode != "no_memory" and self.max_snapshots and len(self.window6) == 6:
            if valence > .20 and continuity > .82:
                self._consolidate(self.p11_fast, self.p01_fast, valence)
        self.arousal, self.valence, self.continuity = arousal, valence, continuity
        self.stats["steps"] += 1
        self.last_step = {
            "step_index": step_index, "active_row": prev_s, "outcome": curr_s,
            "prediction": p_pred, "pre_retrieval_fast": pre_retrieval,
            "after_retrieval_fast": after_retrieval, "pre_update_slow": slow_before,
            "row_buffer_indices": [list(v) for v in self.row_buffer_indices],
            "row_buffer_ages": [[step_index-i for i in v] for v in self.row_buffer_indices],
            "row_buffer_outcomes": [list(v) for v in self.row_buffers],
            "surprise_step": surprise_step, "surprise": self.epistemic_surprise,
            "surprise_trigger": surprise_trigger, "eligibility": eligible,
            "first_step_rejection": blocked, "candidates": candidate_records,
            "accepted_snapshot_index": chosen_index, "accepted_snapshot_probs": chosen_probs,
            "cooldown_before": cooldown_before, "cooldown_at_gate": cooldown_at_gate,
            "cooldown_after": tuple(self.row_cooldown),
            "events": {k: self.stats[k]-stats_before[k] for k in COUNTERS},
        }
        return p_pred

    def _record_candidate(self, reason):
        self.stats["candidate_evaluations"] += 1
        if reason != "fast":
            self.stats["raw_advantage_candidates"] += 1
        self.stats["rejected_"+reason if reason else "gate_passed_candidates"] += 1

    def _consolidate(self, p11, p01, val):
        for p in (p11, p01):
            probability(p)
        finite(val, "valence")
        if not -1 <= val <= 1:
            raise ValueError("valence must be in [-1,1]")
        if self.mode == "no_memory" or not self.max_snapshots:
            return
        self.stats["consolidation_calls"] += 1
        for snap in self.memory:
            if math.hypot(snap.p11-p11, snap.p01-p01) < .12:
                snap.p11 = .5*snap.p11+.5*p11
                snap.p01 = .5*snap.p01+.5*p01
                snap.valence_score = max(snap.valence_score, val)
                self.stats["memory_merges"] += 1
                return
        if len(self.memory) < self.max_snapshots:
            self.memory.append(EpisodicSnapshot(p11, p01, val))
            self.stats["memory_stores"] += 1
        else:
            self.stats["capacity_rejections"] += 1
