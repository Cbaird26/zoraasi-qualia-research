PHASE 2.7 REPRODUCIBLE VERIFICATION — 16 OCTOBER 2026

Start with VERIFICATION_REPORT.html or VERIFICATION_REPORT.txt.
The supplied original script is preserved byte-for-byte. The corrected script
is an importable engine; experiments and deterministic tests live separately.
Frozen Phase 2.5/2.6 source copies are pinned in reference/provenance.json.

Reproduce everything from this folder:
    python3 -B run_all.py

No third-party packages are required. Python 3.9+ syntax is used; the exact
executed interpreter and operating system are in results/reproducibility_run.json.
Only the recorded interpreter/environment was actually tested.
Allow several minutes and about 200 MB of disk for expanded numerical results.
The public corrected_benchmark.json.gz expands automatically when regenerated.
The report builder reads either the uncompressed JSON or its gzip copy.

Individual checks:
    python3 -B original/phase2_7_kl_antirollback_gate.py
    python3 -B corrected/test_phase27.py
    python3 -B review/math_check.py
    python3 -B review/test_benchmark.py
    python3 -B review/audit_original.py
    python3 -B review/benchmark_phase27.py
    python3 -B review/build_report.py

Recorded verification: 49 passing unittest methods; supplied original 100+100
seeds; corrected 30 validation +100 primary +100 follow-up seeds, seven scenarios;
2,800 exact complete forecast-path comparisons against the pinned Phase 2.5 and
its no-memory variant. Validation choices were frozen before test evaluation.
Neither these checks nor a passing unit test require predictive superiority.

Interpretation: gate activity is established, but novel-regime protection fails.
Familiar/early-window gains over no-memory survive, while the complete proposal
has worse overall Brier than Phase 2.5, Phase 2.6 and selected CUSUM on mixed_9.
Overall BCE is better than CUSUM; rankings depend on the scoring rule.
The anti-rollback impossibility claim is refuted by a constructed counterexample.
All results concern functional numerical prediction, not phenomenological
experience or independent physical validation of MQGT-SCF.

Directory guide:
original/   Extracted supplied code and provenance; private bundle also contains
            the uploaded text. The public repository omits that raw conversation.
corrected/  Corrected engine and its deterministic tests.
reference/  Unchanged pinned prior engines and their source hashes/URLs.
review/     Independent mathematics, benchmark, harness checks, notes and diff.
results/    Actual stdout, per-seed numerical output, validation manifest,
            provenance/integrity records and test logs.

SHA256SUMS describes the delivered snapshot. A rerun updates timing/environment
logs and regenerates expanded numerical output, so some checksums will change.
Frozen historical public artifacts are not overwritten by this package.
