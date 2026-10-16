#!/usr/bin/env python3
"""Independent Phase 2.7 algebra, finite-window and causal-contract audit.

Standard library only. Reads original Phase 2.7 and pinned Phase 2.5/2.6 sources;
does not edit them. Candidate compatibility is distinguished from actual retrieval.
The anti-rollback counterexample deliberately stages a legal internal state; it
is not described as a frequency observed from default initialization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import runpy
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
P27 = P26 = P25 = None
FIXTURES = None
A = (0.92, 0.28)
KL_LIMIT = 0.12


def kl(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def counts(path):
    return tuple(sum((left, right) == pair for left, right in zip(path, path[1:]))
                 for pair in ((1, 1), (1, 0), (0, 1), (0, 0)))


def ll(table, p11, p01):
    return sum(n * math.log(p) for n, p in zip(table, (p11, 1 - p11, p01, 1 - p01)))


def row_kl(ones, zeros, snapshot):
    return kl((ones + 0.5) / (ones + zeros + 1), snapshot)


def compatibility(table, snapshot=A):
    n11, n10, n01, n00 = table
    return bool(n11 + n10 and n01 + n00 and
                row_kl(n11, n10, snapshot[0]) <= KL_LIMIT and
                row_kl(n01, n00, snapshot[1]) <= KL_LIMIT)


def weighted_window_enumeration(p11, p01, q0, horizon=6):
    masses = {"total": 0.0, "absent_row": 0.0, "dual_row_kl_compatible": 0.0,
              "exactly_one_row0_transition": 0.0, "singleton_row0_success": 0.0}
    passing_paths = []
    for path in itertools.product((0, 1), repeat=horizon + 1):
        mass = q0 if path[0] else 1 - q0
        for left, right in zip(path, path[1:]):
            p = p11 if left else p01
            mass *= p if right else 1 - p
        table = counts(path)
        n11, n10, n01, n00 = table
        masses["total"] += mass
        if not (n11 + n10 and n01 + n00):
            masses["absent_row"] += mass
        if compatibility(table):
            masses["dual_row_kl_compatible"] += mass
            passing_paths.append({"path": list(path), "counts": list(table), "probability": mass})
        if n01 + n00 == 1:
            masses["exactly_one_row0_transition"] += mass
            if n01 == 1:
                masses["singleton_row0_success"] += mass
    return {"parameters": {"p11": p11, "p01": p01, "q0": q0, "horizon": horizon},
            "candidate": list(A), "probabilities": masses, "compatible_paths": passing_paths,
            "scope": "Coverage and both KL checks only, without fast/slow advantage, distance, arousal, or memory availability."}


def staged_rollback():
    """Five old-A transitions followed by one new-regime transition; old A accepted."""
    engine = P27["Phase27Engine"]()
    engine.p11_fast = engine.p01_fast = 0.5
    engine.p11_slow = engine.p01_slow = 0.5
    engine.arousal = 0.65
    path = (0, 0, 0, 1, 1, 1, 1)
    engine.window = list(zip(path[:5], path[1:6]))
    captured = []
    original_method = engine.candidate_passes_phase27_gate

    def record(self, snap, n11, n10, n01, n00, ll_fast, ll_slow):
        table = (n11, n10, n01, n00)
        passed = original_method(snap, *table, ll_fast, ll_slow)
        captured.append({"passed": passed, "candidate": [snap.p11, snap.p01],
                         "counts": list(table), "ll_fast": ll_fast,
                         "ll_slow_argument": ll_slow,
                         "postupdate_slow_vector": [self.p11_slow, self.p01_slow],
                         "ll_at_postupdate_slow": ll(table, self.p11_slow, self.p01_slow),
                         "distance_used": math.hypot(snap.p11 - self.p11_slow, snap.p01 - self.p01_slow),
                         "row1_kl": row_kl(n11, n10, snap.p11),
                         "row0_kl": row_kl(n01, n00, snap.p01)})
        return passed

    engine.candidate_passes_phase27_gate = types.MethodType(record, engine)
    prediction = engine.step_update(1, 1)
    return {"path": list(path), "old_regime": list(A), "old_transitions": 5,
            "new_transitions": 1, "preupdate_fast": [0.5, 0.5],
            "preupdate_slow": [0.5, 0.5], "preupdate_arousal": 0.65,
            "issued_prediction": prediction, "retrievals_accepted": engine.retrievals_accepted,
            "postupdate_fast": [engine.p11_fast, engine.p01_fast],
            "postupdate_arousal": engine.arousal, "postupdate_continuity": engine.continuity,
            "candidate_evaluations": captured,
            "scope": "Constructed legal state, not a path claimed to arise from default initialization; disproves an unconditional gate guarantee."}


def supplied_antirollback_test_diagnosis():
    table = (4, 1, 0, 1)
    snap_ll, fast_ll, slow_ll = ll(table, *A), ll(table, 0.62, 0.29), ll(table, 0.90, 0.29)
    engine = P27["Phase27Engine"](mode="phase25_ungated")
    snap = P27["EpisodicSnapshot"](*A, 0.5)
    return {"counts": list(table), "snapshot_minus_fast": snap_ll - fast_ll,
            "snapshot_minus_slow": snap_ll - slow_ll,
            "required_margin": 0.30, "row1_kl": row_kl(4, 1, 0.92),
            "ungated_also_rejects": not engine.candidate_passes_phase27_gate(snap, *table, fast_ll, slow_ll)}


def phase26_replay(seed=71000000):
    spec = [(0.92, 0.28, "A"), (0.15, 0.85, "B"), (0.92, 0.28, "A"),
            (0.15, 0.85, "B"), (0.50, 0.50, "C"), (0.88, 0.88, "D"),
            (0.08, 0.12, "E"), (0.15, 0.85, "B"), (0.92, 0.28, "A")]
    schedule = [P25["Block"](45, p11, p01, label) for p11, p01, label in spec]
    fit = P25["generate_training"](seed)
    stream = P25["generate_stream"](schedule, seed, q0=1.0)
    engine = P26["NoveltyAwareEngineV1_2"]()
    engine.initialize(fit.p11, fit.p01, fit.marginal)
    records, predictions = [], []
    for t, (previous, current) in enumerate(zip(stream, stream[1:])):
        retrievals_before = engine.familiar_retrievals
        eligible_before = engine.stats["retrieval_eligible"]
        cooldown_before = engine.cooldown
        prediction = engine.step_update(previous, current)
        predictions.append(prediction)
        block, offset = divmod(t, 45)
        if block and offset < 10:
            records.append({"transition_index_zero_based": t, "block": block,
                            "old_label": spec[block - 1][2], "new_label": spec[block][2],
                            "offset_zero_based": offset, "previous": previous, "current": current,
                            "prediction": prediction, "arousal_after": engine.arousal,
                            "continuity_after": engine.continuity,
                            "retrieved": engine.familiar_retrievals > retrievals_before,
                            "gate_eligible": engine.stats["retrieval_eligible"] > eligible_before,
                            "cooldown_before": cooldown_before, "cooldown_after": engine.cooldown,
                            "window_new_transition_count": min(offset + 1, len(engine.window)),
                            "window_old_transition_count": max(0, len(engine.window) - offset - 1)})
    first3 = [r for r in records if r["offset_zero_based"] < 3]
    affected = sorted({r["block"] for r in first3 if r["retrieved"]})
    scores = P25["score"](stream[1:], predictions)
    prior_recorded_scores = {"brier": 0.18162647823993044, "bce": 0.5487663040857714}
    return {"scope": "One exact prior-primary mixed_9 stream; pinned sources; learned initialization; no estimate of event frequency across other seeds.",
            "seed": seed, "training": asdict(fit), "transitions": len(predictions),
            "stream_sha256": hashlib.sha256(json.dumps(stream, separators=(",", ":")).encode()).hexdigest(),
            "overall_scores": scores,
            "prior_recorded_overall_scores": prior_recorded_scores,
            "prior_recorded_scores_match": scores == prior_recorded_scores,
            "prior_recorded_score_provenance": "Phase 2.6 corrected_benchmark.json, test/mixed_9/seed71000000/phase26_full; independently read during this audit.",
            "switch_count": 8, "switches_with_any_retrieval_in_first3": affected,
            "retrieval_count_at_offset1": sum(r["retrieved"] for r in records if r["offset_zero_based"] == 1),
            "boundary_trace": records,
            "warning": "A retrieval event does not identify the candidate's true regime; no old-regime label is inferred from approximate parameter proximity."}


class MathTests(unittest.TestCase):
    def test_quoted_kl_values_round_correctly(self):
        self.assertAlmostEqual(kl(0.75, 0.28), 0.4745151289838662)
        self.assertAlmostEqual(kl(0.25, 0.28), 0.00228432456344059)

    def test_kl_equals_cross_entropy_minus_entropy(self):
        for p, q in itertools.product((0.25, 0.75, 0.8125), (0.28, 0.92)):
            cross_entropy = -p * math.log(q) - (1 - p) * math.log1p(-q)
            entropy = -p * math.log(p) - (1 - p) * math.log1p(-p)
            self.assertAlmostEqual(kl(p, q), cross_entropy - entropy)
            self.assertGreaterEqual(kl(p, q), 0)

    def test_beta_penalty_is_not_only_negative_log_nplus1(self):
        evidence = math.log(math.factorial(6) * math.factorial(1) / math.factorial(8))
        self.assertAlmostEqual(evidence, -math.log(8) - math.log(7))
        mle_ll = 6 * math.log(6 / 7) + math.log(1 / 7)
        self.assertAlmostEqual(evidence - mle_ll, -1.154537462716286)
        self.assertNotAlmostEqual(evidence - mle_ll, -math.log(8))

    def test_eight_transition_example_and_six_transition_suffix(self):
        path = (1, 1, 1, 1, 0, 1, 1, 1, 1)
        self.assertEqual(counts(path), (6, 1, 1, 0))
        self.assertEqual(counts(path[-7:]), (4, 1, 1, 0))
        self.assertGreater(row_kl(4, 1, 0.92), KL_LIMIT)

    def test_flow_identity_for_all_six_transition_paths(self):
        for path in itertools.product((0, 1), repeat=7):
            n11, n10, n01, n00 = counts(path)
            self.assertEqual(n01 - n10, path[-1] - path[0])

    def test_enumeration_probabilities_normalize(self):
        for result in FIXTURES["window_enumeration"].values():
            self.assertAlmostEqual(result["probabilities"]["total"], 1)

    def test_absent_row_probability_matches_independent_formula(self):
        for result in FIXTURES["window_enumeration"].values():
            p = result["parameters"]
            absence = p["q0"] * p["p11"]**5 + (1 - p["q0"]) * (1 - p["p01"])**5
            self.assertAlmostEqual(absence, result["probabilities"]["absent_row"])

    def test_only_three_six_transition_paths_pass_exact_A_compatibility(self):
        paths = [p for p in itertools.product((0, 1), repeat=7) if compatibility(counts(p))]
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(p[0] == 0 for p in paths))

    def test_true_familiar_rejection_and_novel_compatibility_have_positive_mass(self):
        familiar = FIXTURES["window_enumeration"]["stationary_A"]["probabilities"]
        novel = FIXTURES["window_enumeration"]["stationary_D"]["probabilities"]
        self.assertAlmostEqual(familiar["dual_row_kl_compatible"], 0.076868829184)
        self.assertAlmostEqual(novel["dual_row_kl_compatible"], 0.00877692321792)
        self.assertGreater(1 - familiar["dual_row_kl_compatible"], 0)
        self.assertGreater(novel["dual_row_kl_compatible"], 0)

    def test_single_row0_success_is_rejected_even_when_true_A(self):
        self.assertGreater(row_kl(1, 0, 0.28), KL_LIMIT)
        self.assertLess(row_kl(0, 1, 0.28), KL_LIMIT)
        self.assertGreater(0.28, 0)  # Such a success is possible in familiar A.

    def test_conditional_missing_probability_is_not_all_step_prevalence(self):
        pi = 0.28 / (1 - 0.92 + 0.28)
        self.assertAlmostEqual(1 - pi, 2 / 9)
        self.assertAlmostEqual(pi * 0.08, 0.06222222222222222)
        self.assertNotAlmostEqual(1 - pi, 0.08)
        self.assertAlmostEqual(0.22 * 0.92**2, 0.186208)


class CodeContractTests(unittest.TestCase):
    def test_runtime_window_is_six_not_supplied_test_eight(self):
        engine = P27["Phase27Engine"]()
        path = (1, 1, 1, 1, 0, 1, 1, 1, 1)
        for previous, current in zip(path, path[1:]):
            engine.step_update(previous, current)
        self.assertEqual(len(engine.window), 6)
        self.assertEqual(engine._window_counts(), (4, 1, 1, 0))

    def test_supplied_antirollback_test_already_rejected_by_ungated_margin(self):
        d = FIXTURES["supplied_antirollback_test"]
        self.assertLess(d["snapshot_minus_fast"], d["required_margin"])
        self.assertTrue(d["ungated_also_rejects"])

    def test_full_update_can_retrieve_old_A_with_only_one_new_transition(self):
        d = FIXTURES["staged_rollback"]
        self.assertEqual(d["old_transitions"], 5)
        self.assertEqual(d["new_transitions"], 1)
        self.assertEqual(d["retrievals_accepted"], 1)
        self.assertEqual(d["candidate_evaluations"][0]["candidate"], list(A))

    def test_slow_likelihood_and_distance_use_different_timestamps(self):
        c = FIXTURES["staged_rollback"]["candidate_evaluations"][0]
        self.assertAlmostEqual(c["ll_slow_argument"], ll(c["counts"], 0.5, 0.5))
        self.assertNotEqual(c["ll_slow_argument"], c["ll_at_postupdate_slow"])

    def test_distance_equality_is_accepted(self):
        table = (3, 0, 1, 2)
        radius = math.hypot(0.92 - 0.5, 0.28 - 0.5)
        engine = P27["Phase27Engine"](anti_rollback_dist=radius)
        engine.p11_slow = engine.p01_slow = 0.5
        snapshot = P27["EpisodicSnapshot"](*A, 0.5)
        base = ll(table, 0.5, 0.5)
        self.assertTrue(engine.candidate_passes_phase27_gate(snapshot, *table, base, base))

    def test_marginal_is_scored_after_current_label_update(self):
        engine = P27["Phase27Engine"](mode="no_memory")
        engine.p11_fast = engine.p01_fast = engine.p_marginal_online = 0.5
        engine.p11_slow = engine.p01_slow = 0.5
        engine.step_update(1, 1)
        self.assertAlmostEqual(engine.p_marginal_online, 0.525)
        self.assertAlmostEqual(engine.valence, math.tanh(0.6 * (math.log(0.5) - math.log(0.525))))
        self.assertLess(engine.valence, 0)  # Fair comparison of identical p=.5 forecasts would be zero.

    def test_pinned_phase26_does_not_retrieve_at_every_boundary(self):
        trace = FIXTURES["phase26_replay"]
        self.assertEqual(trace["switches_with_any_retrieval_in_first3"], [7])
        self.assertEqual(trace["retrieval_count_at_offset1"], 0)
        self.assertTrue(trace["prior_recorded_scores_match"])

    def test_actual_phase26_cooldown_reopens_on_sixth_update(self):
        rows = {r["offset_zero_based"]: r for r in FIXTURES["phase26_replay"]["boundary_trace"] if r["block"] == 7}
        self.assertTrue(rows[0]["retrieved"])
        self.assertEqual([rows[t]["cooldown_after"] for t in range(7)], [6, 5, 4, 3, 2, 1, 0])
        self.assertTrue(rows[6]["gate_eligible"])
        self.assertTrue(rows[7]["retrieved"])

    def test_actual_phase26_window_new_transition_count(self):
        rows = [r for r in FIXTURES["phase26_replay"]["boundary_trace"] if r["block"] == 1]
        self.assertEqual((rows[0]["window_old_transition_count"], rows[0]["window_new_transition_count"]), (7, 1))
        self.assertEqual((rows[1]["window_old_transition_count"], rows[1]["window_new_transition_count"]), (6, 2))


def main():
    global P27, P26, P25, FIXTURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "math_check.json")
    args = parser.parse_args()
    P27 = runpy.run_path(str(ROOT / "original" / "phase2_7_kl_antirollback_gate.py"))
    P26 = runpy.run_path(str(ROOT / "reference" / "phase26_engine.py"))
    P25 = runpy.run_path(str(ROOT / "reference" / "phase25_suite.py"))
    FIXTURES = {
        "window_enumeration": {
            "stationary_A": weighted_window_enumeration(0.92, 0.28, 7 / 9),
            "stationary_D": weighted_window_enumeration(0.88, 0.88, 0.88),
            "A_initial_one": weighted_window_enumeration(0.92, 0.28, 1.0),
        },
        "staged_rollback": staged_rollback(),
        "supplied_antirollback_test": supplied_antirollback_test_diagnosis(),
        "phase26_replay": phase26_replay(),
    }
    suites = [unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (MathTests, CodeContractTests)]
    counts_by_class = {cls.__name__: suite.countTestCases() for cls, suite in zip((MathTests, CodeContractTests), suites)}
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(suites))
    if not result.wasSuccessful():
        return 1
    evidence = -math.log(56)
    point = 6 * math.log(0.92) + math.log(0.08)
    report = {"status": "executed", "tests": {"run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "by_class": counts_by_class},
              "quoted_kl": {"KL_0_75_to_0_28": kl(0.75, 0.28), "KL_0_25_to_0_28": kl(0.25, 0.28)},
              "valid_eight_transition_example": {"path": [1,1,1,1,0,1,1,1,1], "counts": [6,1,1,0],
                  "row1_uniform_log_evidence": evidence, "row0_uniform_log_evidence": -math.log(2),
                  "snapshot_A_log_likelihood": point + math.log(0.28),
                  "joint_log_ratio": point + math.log(0.28) - evidence + math.log(2),
                  "evidence_minus_row1_MLE": evidence - (6 * math.log(6/7) + math.log(1/7)),
                  "runtime_trailing_six_counts": [4,1,1,0], "runtime_row1_KL": row_kl(4,1,0.92)},
              **FIXTURES,
              "interpretation": "Finite-window statistical compatibility is not regime identity. Constructed counterexamples refute universal guarantees; one deterministic replay does not identify population causal effects or consciousness."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Tests:", report["tests"])
    print("Quoted KL values:", report["quoted_kl"])
    for name, d in FIXTURES["window_enumeration"].items():
        print(name, d["probabilities"])
    print("Constructed full-update old-A retrieval:", FIXTURES["staged_rollback"]["retrievals_accepted"])
    print("Supplied antirollback test:", FIXTURES["supplied_antirollback_test"])
    print("Pinned P26 one-seed switches with retrieval in first 3 updates:", FIXTURES["phase26_replay"]["switches_with_any_retrieval_in_first3"])
    print("Pinned P26 one-seed score:", FIXTURES["phase26_replay"]["overall_scores"])
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
