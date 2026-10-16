ZoraASI Phase 2.3 / 2.4 executed verification — 15 October 2026

Open verification_report.html for the complete report, or verification_report.txt
for its text equivalent. Actual results, not illustrative numbers, are in results/.

Reproduce everything with Python 3.10+ (standard library only; no network needed):

    python3 -B run_all.py

Run from this directory after extracting the ZIP. Tested runtime: Python 3.14.6
on macOS arm64. The packaged run passed 44 tests: 6 supplied-script tests,
6 public-repository tests, 24 corrected-script tests and 8 independent math tests.
One conformance test additionally checks the public source across 200 seeds.
Runtime in the audited environment was about 25 seconds; other machines vary.
The runner refreshes results/, checks source integrity, and fails on a failed step.

Important results:
- Uploaded Phase 2.3 uses different initialization and derangement than the public
  source. corrected/phase2_3_protocol_reproduction.py defaults to exact source
  behavior; --mode stationary retains the supplied alternative experiment.
- Stationary reset Brier is 0.230072328015; BCE is 0.687342669157, not 0.6869.
- With the public fair initial state, expected reset loss is worse than deranged
  loss. Stationary-reference scores must not be substituted for that experiment.
- Phase 2.4 corrections fix stale state on reinitialization, validation and frozen
  boundary behavior, and add baselines plus causal scalar-state derangement.
- Across switching settings the full engine beats the specified plain EMA, but
  loses to it in stationarity. Removing confidence arbitration improves means.
- All findings concern functional synthetic forecasting, not subjective qualia
  or physical verification of MQGT-SCF.

Contents:
  original/       Both extracted code fences and their provenance checksums.
  corrected/      Separately corrected scripts and regression/causality tests.
  reference/      Unchanged public source snapshot, pinned commit and hashes.
  results/        Actual stdout/stderr, full JSON, per-seed/per-stream scores.
  review/         Independent derivation, Phase 2.4 audit, and code diffs.
  run_all.py      Re-runs original, public, corrected and mathematical checks.
  build_report.py Rebuilds HTML/text presentation from the execution results.
  SHA256SUMS.txt  Integrity manifest for the delivered package (except itself).

The Phase 2.4 extracted filename is original/synthetic_qualia_engine_v1.py;
its embedded docstring calls it phase2_4_functional_state_engine.py. Its code bytes
match the supplied second Python fence. The runner verifies both extracted files against the provenance checksums.
The original raw technical conversation text is excluded from this public copy.

Public reference: Cbaird26/zoraasi-qualia-research
Pinned commit: e941c5fd17a49de71fcdbfbb60cb8d84e168877d
Public repository files are retained with their existing notices and LICENSE.
This public supplement is additive. Existing frozen artifacts are unchanged.
Read INTERPRETATION.txt for implications and falsifiable next experiments.

Monte Carlo mean intervals are exploratory simulation summaries. Holdout seeds
are disjoint, but this audit is not a prospective research preregistration.
Separate agent review was by Codex reviewers, not by external model services.
