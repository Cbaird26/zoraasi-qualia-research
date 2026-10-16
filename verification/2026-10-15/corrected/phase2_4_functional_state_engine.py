#!/usr/bin/env python3
"""Local Phase 2.4 verification extension; original public/uploaded files are unchanged.

Preserves the supplied engine's normal-path dynamics, correcting reset, validation,
and learning-rate-zero behavior. The benchmark uses independent learned Laplace
initialization, simple adaptive baselines, ablations, and causal state derangement.
All named states are functional variables, not measurements of subjective feeling.
BCE is in nats. See review/phase24_notes.txt for design and scientific limitations.
"""

from dataclasses import asdict, dataclass, field, replace
import math
import random
from typing import Dict, List, Tuple
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys


REGIME_A = (0.92, 0.28)
REGIME_B = (0.18, 0.82)


def require_probability(value: float, name: str = "probability") -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def require_binary(value: int, name: str = "observation") -> None:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")


def require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def require_epsilon(eps: float) -> None:
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or not 0 < eps < 0.5:
        raise ValueError("eps must be finite and in (0, 0.5)")


def clip(val: float, low: float, high: float) -> float:
    return min(max(val, low), high)


def binary_entropy_norm(p: float, eps: float = 1e-12) -> float:
    require_probability(p)
    require_epsilon(eps)
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
    p01_fast: float = 0.5
    p_marginal: float = 0.5

    # Slow trace for temporal continuity calculation
    p11_slow: float = 0.5
    p01_slow: float = 0.5

    state: InternalStateVector = field(default_factory=InternalStateVector)
    episodic_memory: List[EpisodicSnapshot] = field(default_factory=list)
    recent_transitions: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_probability(self.fixed_learning_rate, "fixed_learning_rate")
        if isinstance(self.max_memory_slots, bool) or not isinstance(self.max_memory_slots, int) or self.max_memory_slots < 0:
            raise ValueError("max_memory_slots must be a nonnegative integer")
        for name in ("enable_affective_plasticity", "enable_episodic_memory", "enable_self_model"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("p11_fast", "p01_fast", "p_marginal", "p11_slow", "p01_slow"):
            require_probability(getattr(self, name), name)

    def initialize_from_estimates(self, p11_init: float, p01_init: float, p_marg: float) -> None:
        """Reset all run-dependent data; inputs may be learned estimates or labeled oracles."""
        for name, value in (("p11_init", p11_init), ("p01_init", p01_init), ("p_marg", p_marg)):
            require_probability(value, name)
        self.p11_fast = self.p11_slow = p11_init
        self.p01_fast = self.p01_slow = p01_init
        self.p_marginal = p_marg
        self.state = InternalStateVector()
        self.recent_transitions = []
        self.episodic_memory = (
            [EpisodicSnapshot(p11_init, p01_init, valence_at_store=0.5)]
            if self.enable_episodic_memory and self.max_memory_slots else []
        )

    def initialize_from_phase23(self, p11_init: float, p01_init: float, p_marg: float) -> None:
        """Backward-compatible initializer; method name alone does not imply training."""
        self.initialize_from_estimates(p11_init, p01_init, p_marg)

    def predict_proba(self, prev_obs: int) -> float:
        require_binary(prev_obs, "prev_obs")
        p_stateful = self.p11_fast if prev_obs == 1 else self.p01_fast
        if not self.enable_self_model:
            return p_stateful
        # Arbitrate between fast stateful estimate and marginal prior using self-confidence
        w = clip((self.state.self_confidence - 0.5) * 2.0, 0.15, 1.0)
        return w * p_stateful + (1.0 - w) * self.p_marginal

    def step_update(self, prev_obs: int, current_obs: int, eps: float = 1e-12) -> float:
        require_binary(current_obs, "current_obs")
        require_epsilon(eps)
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

        # 3. Update smoothed Brier-complement score (not a calibration metric)
        brier_acc = 1.0 - (current_obs - p_pred) ** 2
        new_conf = clip(0.85 * self.state.self_confidence + 0.15 * brier_acc, 0.0, 1.0)

        # 4. Compute Plasticity / Learning Rate
        if self.enable_affective_plasticity:
            lr = 0.02 + 0.28 * (new_arousal ** 1.5)
        else:
            lr = self.fixed_learning_rate

        # lr=0 must not clip and thereby change a frozen learned estimate.
        if lr > 0:
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


@dataclass(frozen=True)
class TrainingEstimate:
    p11: float
    p01: float
    marginal: float
    n11: int
    n1: int
    n01: int
    n0: int

    def __post_init__(self) -> None:
        for name in ("p11", "p01", "marginal"):
            require_probability(getattr(self, name), name)


def fit_laplace(trials: List[List[int]]) -> TrainingEstimate:
    """Fit Beta(1,1) predictive probabilities from independent training trials."""
    if not trials or any(len(trial) < 2 for trial in trials):
        raise ValueError("Training requires nonempty trials of length at least 2")
    n11 = n1 = n01 = n0 = 0
    for trial in trials:
        for x in trial:
            require_binary(x)
        for prev, y in zip(trial[:-1], trial[1:]):
            if prev == 1:
                n1 += 1
                n11 += y
            else:
                n0 += 1
                n01 += y
    return TrainingEstimate((n11 + 1) / (n1 + 2), (n01 + 1) / (n0 + 2),
                            (n11 + n01 + 1) / (n1 + n0 + 2), n11, n1, n01, n0)


def make_rng(seed: int, namespace: str) -> random.Random:
    """Domain-separated deterministic RNG; stable across Python hash randomization."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    digest = hashlib.sha256(f"phase24-verification-v1|{seed}|{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def generate_training(seed: int, n_trials: int = 24, length: int = 30) -> TrainingEstimate:
    require_positive_int(n_trials, "n_trials")
    require_positive_int(length, "length")
    if length < 2:
        raise ValueError("length must be at least 2")
    rng = make_rng(seed, "independent-training")
    p11, p01 = REGIME_A
    pi = p01 / (1 - p11 + p01)
    trials = []
    for _ in range(n_trials):
        stream = [int(rng.random() < pi)]
        for _ in range(length - 1):
            stream.append(int(rng.random() < (p11 if stream[-1] else p01)))
        trials.append(stream)
    return fit_laplace(trials)


@dataclass
class PlainMarkovEMA:
    """Two transition probabilities, fixed rate, no arbitration or episodic machinery."""
    p11: float
    p01: float
    rate: float = 0.08

    def __post_init__(self) -> None:
        for name in ("p11", "p01", "rate"):
            require_probability(getattr(self, name), name)

    def predict_proba(self, prev_obs: int) -> float:
        require_binary(prev_obs)
        return self.p11 if prev_obs else self.p01

    def step_update(self, prev_obs: int, current_obs: int) -> float:
        require_binary(current_obs)
        pred = self.predict_proba(prev_obs)
        if self.rate > 0:
            value = clip((1 - self.rate) * pred + self.rate * current_obs, 0.02, 0.98)
            if prev_obs:
                self.p11 = value
            else:
                self.p01 = value
        return pred


@dataclass
class MemorylessEMA:
    """Ignores prev_obs at prediction time; optionally updates its marginal estimate."""
    p: float
    rate: float = 0.08

    def __post_init__(self) -> None:
        require_probability(self.p)
        require_probability(self.rate, "rate")

    def predict_proba(self, prev_obs: int) -> float:
        require_binary(prev_obs)
        return self.p

    def step_update(self, prev_obs: int, current_obs: int) -> float:
        require_binary(current_obs)
        pred = self.predict_proba(prev_obs)
        if self.rate > 0:
            self.p = clip((1 - self.rate) * self.p + self.rate * current_obs, 0.02, 0.98)
        return pred


def build_models(fit: TrainingEstimate) -> Dict[str, object]:
    models = {
        "full_engine": SyntheticQualiaEngineV1(),
        "no_episodic_memory": SyntheticQualiaEngineV1(enable_episodic_memory=False),
        "fixed_rate_with_memory": SyntheticQualiaEngineV1(enable_affective_plasticity=False),
        "no_self_model": SyntheticQualiaEngineV1(enable_self_model=False),
        "deranged_state_vector": SyntheticQualiaEngineV1(),
    }
    for model in models.values():
        model.initialize_from_estimates(fit.p11, fit.p01, fit.marginal)
    models.update({
        "plain_fixed_ema": PlainMarkovEMA(fit.p11, fit.p01, 0.08),
        "frozen_learned_markov": PlainMarkovEMA(fit.p11, fit.p01, 0.0),
        "fixed_memoryless": MemorylessEMA(fit.marginal, 0.0),
        "adaptive_memoryless": MemorylessEMA(fit.marginal, 0.08),
    })
    return models


SCENARIOS = ("stationary_a", "periodic_45", "periodic_15", "periodic_90")


def regime_schedule(scenario: str, horizon: int = 360) -> List[Tuple[float, float]]:
    require_positive_int(horizon, "horizon")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}")
    if scenario == "stationary_a":
        return [REGIME_A] * horizon
    duration = int(scenario.rsplit("_", 1)[1])
    return [REGIME_A if (t // duration) % 2 == 0 else REGIME_B for t in range(horizon)]


def generate_stream(schedule: List[Tuple[float, float]], rng: random.Random) -> List[int]:
    if not schedule:
        raise ValueError("schedule must not be empty")
    # Preserve the original Phase 2.4 fixed initial observation. This is not stationarity.
    stream = [1]
    for p11, p01 in schedule:
        require_probability(p11)
        require_probability(p01)
        stream.append(int(rng.random() < (p11 if stream[-1] else p01)))
    return stream


def fixed_derangement(n: int) -> List[int]:
    require_positive_int(n, "n")
    if n < 2:
        raise ValueError("State derangement requires at least two independent streams")
    return [(i + 1) % n for i in range(n)]


def inject_deranged_states(models: List[Dict[str, object]]) -> List[int]:
    """Copy donor full-model vectors BEFORE any current outcome is observed.

    Only the four scalar InternalStateVector fields are transferred. Each recipient's
    fast/slow estimates, memory and recent-transition window remain its own. Donors
    are independent full-engine replicas, not recursively contaminated sham replicas.
    Every vector has the same temporal age; no future observations are consulted.
    """
    sigma = fixed_derangement(len(models))
    snapshots = [replace(group["full_engine"].state) for group in models]
    for i, donor in enumerate(sigma):
        models[i]["deranged_state_vector"].state = replace(snapshots[donor])
    return sigma


def prediction_paths(streams: List[List[int]], fit: TrainingEstimate) -> Dict[str, List[List[float]]]:
    """Prequential predictions, all issued at t before any update with outcomes at t.

    This API receives streams to replay, but predictions only access their prefixes.
    Prefix and current-outcome perturbation tests guard that causal boundary.
    """
    fixed_derangement(len(streams))
    if any(len(s) < 2 for s in streams) or len({len(s) for s in streams}) != 1:
        raise ValueError("streams must have equal lengths of at least 2")
    for stream in streams:
        for value in stream:
            require_binary(value)
    groups = [build_models(fit) for _ in streams]
    paths = {name: [[] for _ in streams] for name in groups[0]}
    for t in range(1, len(streams[0])):
        inject_deranged_states(groups)
        # Pass one: issue ALL predictions before exposing ANY current outcomes.
        for i, group in enumerate(groups):
            for name, model in group.items():
                paths[name][i].append(model.predict_proba(streams[i][t - 1]))
        # Pass two: online updates may use current outcomes for future predictions.
        for i, group in enumerate(groups):
            for name, model in group.items():
                returned = model.step_update(streams[i][t - 1], streams[i][t])
                if returned != paths[name][i][-1]:
                    raise AssertionError("update returned a different pre-observation prediction")
    return paths


def score(y_true: List[int], predictions: List[float]) -> Dict[str, float]:
    if not y_true or len(y_true) != len(predictions):
        raise ValueError("nonempty, equal-length targets and predictions required")
    brier = bce = 0.0
    for y, p in zip(y_true, predictions):
        require_binary(y)
        require_probability(p)
        brier += (p - y) ** 2
        pc = clip(p, 1e-12, 1 - 1e-12)
        bce -= math.log(pc if y else 1 - pc)
    return {"brier": brier / len(y_true), "bce": bce / len(y_true)}


def evaluate_seed(seed: int, scenario: str, n_streams: int = 2, horizon: int = 360) -> dict:
    fixed_derangement(n_streams)
    fit = generate_training(seed)
    schedule = regime_schedule(scenario, horizon)
    streams = [generate_stream(schedule, make_rng(seed, f"test-stream-{i}"))
               for i in range(n_streams)]
    paths = prediction_paths(streams, fit)
    paths["oracle_current_regime"] = [
        [p11 if stream[t] else p01 for t, (p11, p01) in enumerate(schedule)]
        for stream in streams]
    paths["oracle_frozen_regime_a"] = [
        [REGIME_A[0] if x else REGIME_A[1] for x in stream[:-1]] for stream in streams]
    ys = [value for stream in streams for value in stream[1:]]
    metrics = {name: score(ys, [p for stream_probs in values for p in stream_probs])
               for name, values in paths.items()}
    stream_metrics = {name: [score(stream[1:], probabilities)
                            for stream, probabilities in zip(streams, values)]
                      for name, values in paths.items()}
    return {"seed": seed, "scenario": scenario, "fit": asdict(fit),
            "metrics": metrics, "per_stream_metrics": stream_metrics}


def mean_interval(values: List[float]) -> dict:
    if len(values) < 2:
        raise ValueError("At least two seeds are needed for a Monte Carlo mean interval")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(len(values))
    return {"n_seeds": len(values), "mean": mean, "sd_across_seeds": sd,
            "normal_mc_ci95_halfwidth": half, "normal_mc_ci95": [mean - half, mean + half]}


def summarize(records: List[dict]) -> dict:
    summary = {}
    for split in ("primary", "holdout"):
        summary[split] = {}
        for scenario in SCENARIOS:
            rows = [r for r in records if r["split"] == split and r["scenario"] == scenario]
            if not rows:
                continue
            arms = rows[0]["metrics"]
            marginal = {arm: {metric: mean_interval([r["metrics"][arm][metric] for r in rows])
                              for metric in ("brier", "bce")} for arm in arms}
            paired = {
                arm: {metric: mean_interval([
                    r["metrics"]["full_engine"][metric] - r["metrics"][arm][metric] for r in rows])
                      for metric in ("brier", "bce")}
                for arm in arms if arm != "full_engine"}
            summary[split][scenario] = {"metrics": marginal, "paired_full_minus_comparator": paired}
    return summary


def run_benchmark(n_seeds: int = 200, holdout_seeds: int = 100, base_seed: int = 20260921,
                  holdout_base_seed: int = 30260921, n_streams: int = 2, horizon: int = 360) -> dict:
    for value, name in ((n_seeds, "n_seeds"), (holdout_seeds, "holdout_seeds")):
        require_positive_int(value, name)
        if value < 2:
            raise ValueError(f"{name} must be at least 2")
    fixed_derangement(n_streams)
    primary = set(range(base_seed, base_seed + n_seeds))
    holdout = set(range(holdout_base_seed, holdout_base_seed + holdout_seeds))
    if primary & holdout:
        raise ValueError("primary and holdout seeds must be disjoint")
    records = []
    for split, seeds in (("primary", sorted(primary)), ("holdout", sorted(holdout))):
        for scenario in SCENARIOS:
            for seed in seeds:
                record = evaluate_seed(seed, scenario, n_streams, horizon)
                record["split"] = split
                records.append(record)
    return {
        "schema": "phase24-local-verification-v1",
        "metadata": {
            "python": sys.version, "platform": platform.platform(),
            "seeds_primary": [base_seed, base_seed + n_seeds - 1],
            "seeds_holdout": [holdout_base_seed, holdout_base_seed + holdout_seeds - 1],
            "n_primary": n_seeds, "n_holdout": holdout_seeds, "n_streams_per_seed": n_streams,
            "horizon": horizon, "training": "24 independent stationary-A trials of length 30; Beta(1,1)",
            "rng": "SHA256 domain-separated independent training/test streams; common random numbers across scenarios",
            "initial_observation": 1, "scenario_names": SCENARIOS,
            "sham_mapping": fixed_derangement(n_streams),
            "sham_scope": "Only four scalar state-vector fields; target transition estimates/history/memory retained",
            "sham_causality": "Donor full-engine state at t-1 copied before any stream is updated at t",
            "sham_limitation": "Independent observation streams share regime timing; not an unconditional independence null",
            "ci": "Normal approximation mean +/- 1.96*s/sqrt(N); seeds are the units, not transitions or streams",
            "paired_sign": "full engine loss minus comparator loss; negative favors full engine",
            "holdout": "Disjoint fixed seeds; no hyperparameter search. Exploratory robustness, not independent preregistration.",
            "interpretation": "Functional prediction/state adaptation only; no phenomenological or MQGT-SCF validation",
        }, "summary": summarize(records), "raw_per_seed": records,
    }


def print_summary(report: dict) -> None:
    print("Phase 2.4 corrected local benchmark; BCE in nats")
    print("Intervals: normal Monte Carlo 95% mean intervals across independent seed replicates")
    print("Paired differences: full engine minus comparator; negative favors full engine")
    for split, scenarios in report["summary"].items():
        for scenario, data in scenarios.items():
            print(f"\n{split} / {scenario}")
            print(f"{'Model':29} {'Brier mean +/- CI':23} {'BCE mean +/- CI':23}")
            for arm, metrics in data["metrics"].items():
                bs, ce = metrics["brier"], metrics["bce"]
                print(f"{arm:29} {bs['mean']:.6f} +/- {bs['normal_mc_ci95_halfwidth']:.6f}  "
                      f"{ce['mean']:.6f} +/- {ce['normal_mc_ci95_halfwidth']:.6f}")
            print("Paired full-minus-comparator:")
            for arm in ("plain_fixed_ema", "no_episodic_memory", "no_self_model", "deranged_state_vector"):
                bs, ce = (data["paired_full_minus_comparator"][arm][metric] for metric in ("brier", "bce"))
                print(f"  {arm:27} {bs['mean']:+.6f} +/- {bs['normal_mc_ci95_halfwidth']:.6f}  "
                      f"{ce['mean']:+.6f} +/- {ce['normal_mc_ci95_halfwidth']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--holdout-seeds", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20260921)
    parser.add_argument("--holdout-base-seed", type=int, default=30260921)
    parser.add_argument("--streams", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=360)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(args.seeds, args.holdout_seeds, args.base_seed,
                           args.holdout_base_seed, args.streams, args.horizon)
    print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"\nWrote reproducible per-seed results to {args.output}")


if __name__ == "__main__":
    main()

