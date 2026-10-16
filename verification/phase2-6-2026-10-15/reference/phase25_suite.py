#!/usr/bin/env python3
"""Phase 2.5 local verification/falsification suite. Standard library only.

Functional prediction/state adaptation only: no phenomenological inference.
Original artifacts remain separate. Learned initialization, independent validation
selection, causal donor controls, random-memory ablation, paired seed-level scores.
"""
from dataclasses import dataclass, field, asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from typing import Dict, List, Optional, Tuple

A = (.92, .28)
B = (.15, .85)  # supplied Phase 2.5 B; differs from Phase 2.4 (.18,.82)
COUNTERS = ("retrieval_eligible", "candidate_draws", "candidate_is_best", "retrievals",
            "consolidation_calls", "snapshot_merges", "snapshot_stores", "capacity_rejections")


def probability(p, name="probability"):
    if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")


def binary(x):
    if x not in (0, 1):
        raise ValueError("observation must be 0 or 1")


def positive_int(n, name):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_scalars(values):
    if len(values) != 3:
        raise ValueError("donor scalars must be (arousal, valence, continuity)")
    probability(values[0], "arousal")
    probability(values[2], "continuity")
    if not math.isfinite(values[1]) or not -1 <= values[1] <= 1:
        raise ValueError("valence must be finite and in [-1,1]")


def clip(value, low, high):
    return min(max(value, low), high)


def make_rng(seed: int, namespace: str):
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    digest = hashlib.sha256(f"phase25-local-v1|{seed}|{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def score(y_true, predictions):
    if not y_true or len(y_true) != len(predictions):
        raise ValueError("nonempty equal-length targets and predictions required")
    bs = ce = 0.
    for y, p in zip(y_true, predictions):
        binary(y)
        probability(p)
        bs += (y-p)**2
        pc = clip(p, 1e-12, 1-1e-12)
        ce -= math.log(pc if y else 1-pc)
    return {"brier": bs/len(y_true), "bce": ce/len(y_true)}


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
        retrieval_mode: str = "best",
        rng_seed: int = 0,
        legacy_marginal_order: bool = False,
    ) -> None:
        for value, name in ((fixed_lr, "fixed_lr"),):
            probability(value, name)
        if not math.isfinite(retrieval_ll_margin) or retrieval_ll_margin < 0:
            raise ValueError("retrieval_ll_margin must be nonnegative and finite")
        if isinstance(max_snapshots, bool) or not isinstance(max_snapshots, int) or max_snapshots < 0:
            raise ValueError("max_snapshots must be a nonnegative integer")
        if retrieval_mode not in ("best", "random"):
            raise ValueError("retrieval_mode must be best or random")
        for flag in (enable_episodic_memory, enable_affective_plasticity, legacy_marginal_order):
            if not isinstance(flag, bool):
                raise ValueError("flags must be boolean")
        self.enable_episodic_memory = enable_episodic_memory
        self.enable_affective_plasticity = enable_affective_plasticity
        self.fixed_lr = fixed_lr
        self.retrieval_ll_margin = retrieval_ll_margin
        self.max_snapshots = max_snapshots
        self.retrieval_mode = retrieval_mode
        self.rng_seed = rng_seed
        self.legacy_marginal_order = legacy_marginal_order  # audit-only bug recreation
        self.initialize(.5, .5, .5)

    def initialize(self, p11: float, p01: float, marginal: float) -> None:
        for value in (p11, p01, marginal):
            probability(value)
        self.p11_fast = self.p11_slow = p11
        self.p01_fast = self.p01_slow = p01
        self.p_marginal_online = marginal
        self.arousal, self.valence, self.continuity = .20, 0., 1.
        self.memory = ([EpisodicSnapshot(p11, p01, .5)]
                       if self.enable_episodic_memory and self.max_snapshots else [])
        self.window = []
        self.recorded_trace = ScalarTrace()
        self.rng = make_rng(self.rng_seed, "snapshot-candidate")
        self.stats = {name: 0 for name in COUNTERS}
        self.last_candidate_index = self.last_best_index = None

    def scalars(self) -> Tuple[float, float, float]:
        return self.arousal, self.valence, self.continuity

    def predict_proba(self, prev_s: int) -> float:
        binary(prev_s)
        return self.p11_fast if prev_s == 1 else self.p01_fast

    def step_update(
        self,
        prev_s: int,
        curr_s: int,
        donor_scalars: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        binary(curr_s)
        if donor_scalars is not None:
            validate_scalars(donor_scalars)
        p_pred = self.predict_proba(prev_s)
        err = abs(curr_s - p_pred)

        # Online marginal baseline for valence comparison (avoids stale Regime A bias)
        marginal_before = self.p_marginal_online
        if self.legacy_marginal_order:
            self.p_marginal_online = .95 * marginal_before + .05 * curr_s
        pc = clip(p_pred, 1e-12, 1.0 - 1e-12)
        pm = clip(self.p_marginal_online, 1e-12, 1.0 - 1e-12)
        loss_agent = -(curr_s * math.log(pc) + (1 - curr_s) * math.log(1.0 - pc))
        loss_marg = -(curr_s * math.log(pm) + (1 - curr_s) * math.log(1.0 - pm))
        # Both losses now compare predictions made before observing curr_s.
        self.p_marginal_online = .95 * marginal_before + .05 * curr_s

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

        if lr > 0:  # zero rate must not change an extreme frozen estimate by clipping
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
                self.stats["retrieval_eligible"] += 1
                likelihoods = [self._window_ll(s.p11, s.p01) for s in self.memory]
                best_index = max(range(len(self.memory)), key=likelihoods.__getitem__)
                # Draw in BOTH modes to match RNG consumption when eligibility matches.
                random_index = self.rng.randrange(len(self.memory))
                self.stats["candidate_draws"] += 1
                chosen = random_index if self.retrieval_mode == "random" else best_index
                self.last_candidate_index, self.last_best_index = chosen, best_index
                self.stats["candidate_is_best"] += int(chosen == best_index)
                best_snap = self.memory[chosen]
                curr_ll = self._window_ll(self.p11_fast, self.p01_fast)
                if likelihoods[chosen] > curr_ll + self.retrieval_ll_margin:
                    self.stats["retrievals"] += 1
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
        self.stats["consolidation_calls"] += 1
        for snap in self.memory:
            if math.hypot(snap.p11 - p11, snap.p01 - p01) < 0.12:
                snap.p11 = 0.5 * snap.p11 + 0.5 * p11
                snap.p01 = 0.5 * snap.p01 + 0.5 * p01
                snap.valence_score = max(snap.valence_score, val)
                self.stats["snapshot_merges"] += 1
                return
        if len(self.memory) < self.max_snapshots:
            self.memory.append(EpisodicSnapshot(p11, p01, val))
            self.stats["snapshot_stores"] += 1
        else:
            self.stats["capacity_rejections"] += 1


@dataclass(frozen=True)
class TrainingEstimate:
    p11: float
    p01: float
    marginal: float
    n11: int
    n1: int
    n01: int
    n0: int
    training_seed: int


def generate_training(seed: int) -> TrainingEstimate:
    training_seed = seed + 1_000_000_000
    rng = make_rng(training_seed, "independent-training-observations")
    n11 = n1 = n01 = n0 = 0
    pi = A[1] / (1 - A[0] + A[1])
    for _ in range(24):
        prev = int(rng.random() < pi)
        for _ in range(29):
            y = int(rng.random() < (A[0] if prev else A[1]))
            if prev:
                n1 += 1
                n11 += y
            else:
                n0 += 1
                n01 += y
            prev = y
    return TrainingEstimate((n11+1)/(n1+2), (n01+1)/(n0+2),
                            (n11+n01+1)/(n1+n0+2), n11, n1, n01, n0, training_seed)


@dataclass
class PlainEMA:
    p11: float
    p01: float
    rate: float

    def __post_init__(self):
        for value in (self.p11, self.p01, self.rate):
            probability(value)

    def predict_proba(self, prev):
        binary(prev)
        return self.p11 if prev else self.p01

    def step_update(self, prev, y):
        binary(y)
        p = self.predict_proba(prev)
        if self.rate > 0:
            value = clip((1-self.rate)*p + self.rate*y, .02, .98)
            if prev:
                self.p11 = value
            else:
                self.p01 = value
        return p


class ErrorCUSUM:
    """Fixed-reference absolute-error CUSUM + EMA/reset; not classic Page-Hinkley."""
    def __init__(self, p11, p01, rate=.10, threshold=1.65, reference=.38):
        for value in (p11, p01, rate, reference):
            probability(value)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive and finite")
        self.p11, self.p01, self.rate = p11, p01, rate
        self.threshold, self.reference = threshold, reference
        self.cum_err = self.min_cum_err = 0.
        self.resets = 0

    def predict_proba(self, prev):
        binary(prev)
        return self.p11 if prev else self.p01

    def step_update(self, prev, y):
        binary(y)
        p = self.predict_proba(prev)
        self.cum_err += abs(y-p) - self.reference
        self.min_cum_err = min(self.min_cum_err, self.cum_err)
        if self.cum_err-self.min_cum_err > self.threshold:
            self.p11 = self.p01 = .5
            self.cum_err = self.min_cum_err = 0.
            self.resets += 1
        if prev:
            self.p11 = (1-self.rate)*self.p11 + self.rate*y
        else:
            self.p01 = (1-self.rate)*self.p01 + self.rate*y
        return p


@dataclass(frozen=True)
class Block:
    dwell: int
    p11: float
    p01: float
    label: str

    def __post_init__(self):
        positive_int(self.dwell, "dwell")
        probability(self.p11)
        probability(self.p01)


def flatten_schedule(schedule):
    if not schedule:
        raise ValueError("schedule must not be empty")
    return [(b.p11, b.p01) for b in schedule for _ in range(b.dwell)]


def periodic_schedule(horizon=450, duration=45, offset=0, family=(A, B)):
    """offset=23 gives initial A22, then B45...; final A23 for horizon=360."""
    positive_int(horizon, "horizon")
    positive_int(duration, "duration")
    if not isinstance(offset, int):
        raise ValueError("offset must be an integer")
    points = [((t+offset)//duration) % 2 for t in range(horizon)]
    blocks = []
    for regime in points:
        p11, p01 = family[regime]
        label = "A" if regime == 0 else "B"
        if blocks and blocks[-1].label == label:
            old = blocks.pop()
            blocks.append(Block(old.dwell+1, p11, p01, label))
        else:
            blocks.append(Block(1, p11, p01, label))
    return blocks


def stochastic_schedule(seed, namespace, family=(A, B), horizon=450):
    positive_int(horizon, "horizon")
    rng = make_rng(seed, namespace)
    blocks, total = [], 0
    regime = int(rng.random() < .5)
    while total < horizon:
        dwell = min(rng.randint(25, 65), horizon-total)
        p11, p01 = family[regime]
        blocks.append(Block(dwell, p11, p01, "A" if regime == 0 else "B"))
        total += dwell
        regime = 1-regime
    return blocks


SCENARIOS = ("recurrent_45", "familiar_novel", "unseen_random", "stationary_a",
             "recurrent_q0_zero", "recurrent_q0_half")


def make_scenario(name, seed):
    if name in ("recurrent_45", "recurrent_q0_zero", "recurrent_q0_half"):
        q0 = 0. if name.endswith("zero") else .5 if name.endswith("half") else 1.
        return periodic_schedule(), q0
    if name == "familiar_novel":
        sequence = [("A", A), ("B", B), ("A", A), ("B", B),
                    ("C", (.5, .5)), ("D", (.88, .88)), ("B", B), ("A", A)]
        return [Block(45, *params, label) for label, params in sequence], 1.
    if name == "unseen_random":
        return stochastic_schedule(seed, "recipient-schedule", ((.80, .12), (.30, .70))), 1.
    if name == "stationary_a":
        return [Block(450, *A, "A")], 1.
    raise ValueError(f"unknown scenario {name!r}")


def generate_stream(schedule, seed, namespace="recipient-observations", q0=1.):
    probability(q0, "initial-state probability")
    rng = make_rng(seed, namespace)
    # Consume a separate initial draw even for q0=0/1, preserving paired innovations.
    stream = [int(rng.random() < q0)]
    for p11, p01 in flatten_schedule(schedule):
        stream.append(int(rng.random() < (p11 if stream[-1] else p01)))
    return stream


def rotate_schedule(schedule, shift=23):
    points = [(b.p11, b.p01, b.label) for b in schedule for _ in range(b.dwell)]
    shift %= len(points)
    points = points[shift:] + points[:shift]
    blocks = []
    for p11, p01, label in points:
        if blocks and (blocks[-1].p11, blocks[-1].p01, blocks[-1].label) == (p11, p01, label):
            old = blocks.pop()
            blocks.append(Block(old.dwell+1, p11, p01, label))
        else:
            blocks.append(Block(1, p11, p01, label))
    return blocks


@dataclass(frozen=True)
class SelectedBaselines:
    ema_rate: float
    cusum_rate: float
    cusum_threshold: float
    cusum_reference: float


def baseline_predictions(model, stream):
    return [model.step_update(prev, y) for prev, y in zip(stream[:-1], stream[1:])]


def tune_baselines(n_validation=30, validation_base=41000000):
    """No test seeds, data, or evaluation results are accepted by this function."""
    positive_int(n_validation, "n_validation")
    ema_grid = [i/100 for i in range(1, 51)]
    cusum_grid = [(rate, threshold, reference)
                 for rate in (.02, .05, .10, .20, .35, .50)
                 for threshold in (.50, 1., 1.65, 2.5, 4., 1e9)
                 for reference in (.30, .38, .45)]
    datasets = []
    for seed in range(validation_base, validation_base+n_validation):
        fit = generate_training(seed)
        schedule, q0 = make_scenario("recurrent_45", seed)
        datasets.append((seed, fit, generate_stream(schedule, seed, q0=q0)))
    tables = {"ema": [], "cusum": []}
    for family, grid in (("ema", ema_grid), ("cusum", cusum_grid)):
        for config in grid:
            raw = []
            for seed, fit, stream in datasets:
                model = (PlainEMA(fit.p11, fit.p01, config) if family == "ema" else
                         ErrorCUSUM(fit.p11, fit.p01, *config))
                raw.append({"seed": seed, **score(stream[1:], baseline_predictions(model, stream))})
            params = ({"rate": config} if family == "ema" else
                      dict(zip(("rate", "threshold", "reference"), config)))
            tables[family].append({"params": params, "mean_brier": statistics.mean(r["brier"] for r in raw),
                                   "mean_bce": statistics.mean(r["bce"] for r in raw), "per_seed": raw})
    # Stable iteration order breaks exact ties; only validation Brier selects models.
    best_ema = min(tables["ema"], key=lambda r: r["mean_brier"])["params"]
    best_cusum = min(tables["cusum"], key=lambda r: r["mean_brier"])["params"]
    selected = SelectedBaselines(best_ema["rate"], best_cusum["rate"],
                                 best_cusum["threshold"], best_cusum["reference"])
    return {"stage": "validation_only_completed_before_test", "selection_objective": "mean validation Brier",
            "validation_scenario": "recurrent_45", "validation_seeds": [validation_base, validation_base+n_validation-1],
            "training_seed_offset": 1_000_000_000, "selected": asdict(selected), "candidate_results": tables,
            "status": "Exploratory fixed design, not independent preregistration; same selected models used for all test scenarios."}


def build_models(fit, selected, seed):
    configs = {
        "full_engine": {}, "no_memory": {"enable_episodic_memory": False},
        "random_snapshot": {"retrieval_mode": "random"},
        "donor_sync": {}, "donor_offset": {}, "donor_independent": {},
        "legacy_marginal_order": {"legacy_marginal_order": True},
    }
    models = {}
    for name, config in configs.items():
        model = RefinedQualiaEngineV1_1(rng_seed=seed, **config)
        model.initialize(fit.p11, fit.p01, fit.marginal)
        models[name] = model
    models["validation_tuned_ema"] = PlainEMA(fit.p11, fit.p01, selected.ema_rate)
    models["validation_tuned_cusum"] = ErrorCUSUM(fit.p11, fit.p01, selected.cusum_rate,
                                                selected.cusum_threshold, selected.cusum_reference)
    models["frozen_learned_markov"] = PlainEMA(fit.p11, fit.p01, 0.)
    return models


def run_paths(stream, donor_streams, fit, selected, seed=0):
    """All current predictions precede current observations; donor t affects t+1.

    Donor scalars are same-time POST-observation updates, matching endogenous scalar
    timing. They never alter the already-scored current prediction. This differs
    deliberately from Phase 2.4's pre-update vector copy protocol.
    """
    if len(stream) < 2 or set(donor_streams) != {"sync", "offset", "independent"}:
        raise ValueError("nonempty target and exactly three named donor streams required")
    if any(len(s) != len(stream) for s in donor_streams.values()):
        raise ValueError("all donor streams must match target length")
    for s in [stream, *donor_streams.values()]:
        for x in s:
            binary(x)
    models = build_models(fit, selected, seed)
    donors = {}
    for name in donor_streams:
        donors[name] = RefinedQualiaEngineV1_1(rng_seed=seed)
        donors[name].initialize(fit.p11, fit.p01, fit.marginal)
    paths = {name: [] for name in models}
    stats_trace = {name: [] for name, model in models.items() if hasattr(model, "stats")}
    for t in range(1, len(stream)):
        for name, model in models.items():
            paths[name].append(model.predict_proba(stream[t-1]))
        # Even donor predictions are issued before any current outcomes are exposed.
        donor_predictions = {name: model.predict_proba(donor_streams[name][t-1])
                             for name, model in donors.items()}
        for name, donor in donors.items():
            returned = donor.step_update(donor_streams[name][t-1], donor_streams[name][t])
            if returned != donor_predictions[name]:
                raise AssertionError("donor current prediction changed during update")
        for name, model in models.items():
            if name.startswith("donor_"):
                returned = model.step_update(stream[t-1], stream[t], donors[name[6:]].scalars())
            else:
                returned = model.step_update(stream[t-1], stream[t])
            if returned != paths[name][-1]:
                raise AssertionError("recipient current prediction changed during update")
            if name in stats_trace:
                stats_trace[name].append({**model.stats, "memory_size": len(model.memory)})
    return paths, stats_trace, {"cusum_resets": models["validation_tuned_cusum"].resets}


def replay_trace(stream, scalar_trace, fit, seed=0):
    if len(stream) < 2 or len(scalar_trace) != len(stream)-1:
        raise ValueError("trace must have one scalar tuple per transition")
    for values in scalar_trace:
        validate_scalars(values)
    model = RefinedQualiaEngineV1_1(rng_seed=seed)
    model.initialize(fit.p11, fit.p01, fit.marginal)
    return [model.step_update(stream[t-1], stream[t], scalar_trace[t-1])
            for t in range(1, len(stream))]


def indexes_and_blocks(schedule):
    scopes = {"overall": [], "initial_30": [], "first_visits": [], "revisited": [],
              "early_10_after_change": [], "later_than_10": [], "novel_C_D": [], "postnovel_revisited": []}
    blocks, seen, start, after_novel = [], {}, 0, False
    for index, b in enumerate(schedule):
        end = start+b.dwell
        occurrence = seen.get(b.label, 0)+1
        seen[b.label] = occurrence
        block = {"index": index, "label": b.label, "start": start, "end": end,
                 "dwell": b.dwell, "p11": b.p11, "p01": b.p01, "occurrence_in_test": occurrence}
        blocks.append(block)
        scopes["overall"].extend(range(start, end))
        scopes["first_visits" if occurrence == 1 else "revisited"].extend(range(start, end))
        if index > 0:
            scopes["early_10_after_change"].extend(range(start, min(start+10, end)))
        scopes["later_than_10"].extend(range(min(start+10, end), end))
        if b.label in ("C", "D"):
            scopes["novel_C_D"].extend(range(start, end))
            after_novel = True
        elif after_novel and occurrence > 1:
            scopes["postnovel_revisited"].extend(range(start, end))
        start = end
    scopes["initial_30"] = list(range(min(30, start)))
    return {k: v for k, v in scopes.items() if v}, blocks


def evaluate_seed(seed, scenario, selected):
    schedule, q0 = make_scenario(scenario, seed)
    fit = generate_training(seed)
    stream = generate_stream(schedule, seed, q0=q0)
    family = ((.80, .12), (.30, .70)) if scenario == "unseen_random" else (A, B)
    schedules = {"sync": schedule, "offset": rotate_schedule(schedule),
                 "independent": stochastic_schedule(seed, "independent-donor-schedule", family, len(stream)-1)}
    donor_streams = {name: generate_stream(sch, seed, f"donor-{name}-observations", q0)
                    for name, sch in schedules.items()}
    paths, stats_trace, additional = run_paths(stream, donor_streams, fit, selected, seed)
    paths["oracle_current_regime"] = [p11 if stream[t] else p01
                                      for t, (p11, p01) in enumerate(flatten_schedule(schedule))]
    scopes, blocks = indexes_and_blocks(schedule)
    targets = stream[1:]
    metrics = {scope: {name: score([targets[i] for i in indices], [predictions[i] for i in indices])
                       for name, predictions in paths.items()} for scope, indices in scopes.items()}
    for block in blocks:
        start, end = block["start"], block["end"]
        block["metrics"] = {name: score(targets[start:end], p[start:end]) for name, p in paths.items()}
        block["first_10_metrics"] = {name: score(targets[start:min(start+10, end)], p[start:min(start+10, end)])
                                     for name, p in paths.items()}
        block["memory_events"] = {}
        for name, trace in stats_trace.items():
            before = trace[start-1] if start else dict.fromkeys(COUNTERS, 0)
            block["memory_events"][name] = {key: trace[end-1][key]-before[key] for key in COUNTERS}
            block["memory_events"][name]["memory_size_at_end"] = trace[end-1]["memory_size"]
    return {"seed": seed, "scenario": scenario, "q0": q0, "training": asdict(fit), "metrics": metrics,
            "blocks": blocks, "donor_schedules": {k: [asdict(b) for b in v] for k, v in schedules.items()},
            "events": {name: trace[-1] for name, trace in stats_trace.items()}, **additional}


def mean_ci(values):
    if len(values) < 2:
        raise ValueError("at least two independent seed replicates required for mean intervals")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = 1.96*sd/math.sqrt(len(values))
    return {"n_seeds": len(values), "mean": mean, "sd": sd, "normal_mc_ci95_halfwidth": half,
            "normal_mc_ci95": [mean-half, mean+half]}


def summarize(records):
    summary = {}
    for split in ("test", "followup"):
        summary[split] = {}
        for scenario in SCENARIOS:
            rows = [r for r in records if r["split"] == split and r["scenario"] == scenario]
            if not rows:
                continue
            scopes = {}
            for scope, arm_dict in rows[0]["metrics"].items():
                scopes[scope] = {
                    "metrics": {arm: {metric: mean_ci([r["metrics"][scope][arm][metric] for r in rows])
                                      for metric in ("brier", "bce")} for arm in arm_dict},
                    "paired_full_minus_comparator": {
                        arm: {metric: mean_ci([r["metrics"][scope]["full_engine"][metric]-r["metrics"][scope][arm][metric]
                                               for r in rows]) for metric in ("brier", "bce")}
                        for arm in arm_dict if arm != "full_engine"}}
            events = {arm: {key: mean_ci([r["events"][arm][key] for r in rows])
                            for key in rows[0]["events"][arm]} for arm in rows[0]["events"]}
            summary[split][scenario] = {"scopes": scopes, "events": events}
    return summary


def check_seed_sets(n_validation, n_test, n_followup, validation_base, test_base, followup_base):
    for n in (n_validation, n_test, n_followup):
        positive_int(n, "number of seeds")
        if n < 2:
            raise ValueError("at least two seeds per split required")
    sets = [set(range(base, base+n)) for base, n in
            ((validation_base, n_validation), (test_base, n_test), (followup_base, n_followup))]
    for i in range(3):
        for j in range(i+1, 3):
            if sets[i] & sets[j]:
                raise ValueError("validation/test/followup seed ranges must be disjoint")
    observations = set.union(*sets)
    training = {s+1_000_000_000 for s in observations}
    if observations & training:
        raise ValueError("training and evaluation seeds overlap")


def run_suite(output, n_test=100, n_validation=30, n_followup=100, validation_base=41000000,
              test_base=51000000, followup_base=61000000):
    check_seed_sets(n_validation, n_test, n_followup, validation_base, test_base, followup_base)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = tune_baselines(n_validation, validation_base)
    manifest["implementation_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["evaluation_plan_before_test"] = {"test_seeds": [test_base, test_base+n_test-1],
        "followup_seeds": [followup_base, followup_base+n_followup-1], "scenarios": SCENARIOS,
        "primary_score": "Brier", "secondary_score": "BCE nats", "paired_sign": "full minus comparator",
        "training": "24 independent stationary-A trials, 30 states, Beta(1,1) smoothing"}
    manifest_path = output.with_name("corrected_validation_manifest.json")
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False)+"\n").encode()
    manifest_path.write_bytes(manifest_bytes)  # completed and saved BEFORE test data/evaluation
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    # Re-read the saved choices; test evaluation cannot retune them.
    selected = SelectedBaselines(**json.loads(manifest_path.read_text())["selected"])
    print(f"Validation complete; frozen choices saved to {manifest_path}", flush=True)
    print(f"Manifest SHA256: {digest}\nSelected: {asdict(selected)}", flush=True)
    records = []
    for split, base, count in (("test", test_base, n_test), ("followup", followup_base, n_followup)):
        for scenario in SCENARIOS:
            for seed in range(base, base+count):
                row = evaluate_seed(seed, scenario, selected)
                row["split"] = split
                records.append(row)
            print(f"Completed {split} / {scenario} / {count} seeds", flush=True)
    report = {"schema": "phase25-local-verification-v1", "metadata": {
        "python": sys.version, "platform": platform.platform(), "validation_manifest_sha256": digest,
        "selected_baselines": asdict(selected), "training_seed_offset": 1_000_000_000,
        "validation_seed_range": [validation_base, validation_base+n_validation-1],
        "test_seed_range": [test_base, test_base+n_test-1],
        "followup_seed_range": [followup_base, followup_base+n_followup-1],
        "ci": "mean +/-1.96*sample SD/sqrt(N); independent seeds are sampling units, not blocks/steps",
        "multiple_testing": "exploratory pointwise intervals; no multiplicity adjustment",
        "donor_timing": "all predictions at t precede all observations at t; donor scalars after t only affect t+1 onward",
        "offset_control": "fixed 23-transition circular rotation; independent timing uses separate stochastic dwell draws",
        "random_control": "uniform candidate index under same eligibility and likelihood gate; same per-seed candidate RNG",
        "scope": "functional simulation only; no phenomenological or physical-theory validation",
        "status": "Executed exploratory verification; disjoint followup seeds are not independent preregistration"},
        "summary": summarize(records), "raw_per_seed": records}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print_summary(report)
    return report


def print_summary(report):
    print("\nScores: means +/- normal 95% Monte Carlo interval halfwidth; BCE in nats")
    for split, scenarios in report["summary"].items():
        for scenario, data in scenarios.items():
            print(f"\n{split} / {scenario}")
            scope = data["scopes"]["overall"]
            for arm, metrics in scope["metrics"].items():
                bs, ce = metrics["brier"], metrics["bce"]
                print(f"{arm:27} {bs['mean']:.6f} +/- {bs['normal_mc_ci95_halfwidth']:.6f}  "
                      f"{ce['mean']:.6f} +/- {ce['normal_mc_ci95_halfwidth']:.6f}")
            for subset in ("overall", "novel_C_D", "postnovel_revisited", "early_10_after_change", "initial_30"):
                if subset not in data["scopes"]:
                    continue
                print(f"Paired full-minus-comparator / {subset}:")
                for arm in ("validation_tuned_ema", "validation_tuned_cusum", "no_memory", "random_snapshot",
                            "donor_sync", "donor_offset", "donor_independent", "legacy_marginal_order"):
                    metrics = data["scopes"][subset]["paired_full_minus_comparator"][arm]
                    bs, ce = metrics["brier"], metrics["bce"]
                    print(f"  {arm:25} {bs['mean']:+.6f} +/- {bs['normal_mc_ci95_halfwidth']:.6f}  "
                          f"{ce['mean']:+.6f} +/- {ce['normal_mc_ci95_halfwidth']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-seeds", type=int, default=100)
    parser.add_argument("--validation-seeds", type=int, default=30)
    parser.add_argument("--followup-seeds", type=int, default=100)
    parser.add_argument("--validation-base", type=int, default=41000000)
    parser.add_argument("--test-base", type=int, default=51000000)
    parser.add_argument("--followup-base", type=int, default=61000000)
    parser.add_argument("--output", type=Path, default=Path("results/corrected_benchmark.json"))
    args = parser.parse_args()
    run_suite(args.output, args.test_seeds, args.validation_seeds, args.followup_seeds,
              args.validation_base, args.test_base, args.followup_base)


if __name__ == "__main__":
    main()

