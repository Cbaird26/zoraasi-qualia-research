PHASE 2.8 REPRODUCIBLE VERIFICATION — 16 OCTOBER 2026

Start with VERIFICATION_REPORT.html or VERIFICATION_REPORT.txt.
The original script is preserved byte-for-byte. The corrected script is an
importable engine; experiment runners and tests are separate. All prior source
copies are pinned to commit 669037f in reference/provenance.json.

Reproduce everything from this folder:
    python3 -B run_all.py

Standard-library Python only. The exact interpreter and system used are recorded
in results/reproducibility_run.json; other environments were not tested. Allow
several minutes and about 250 MB for expanded results. The public compressed
corrected_benchmark.json.gz regenerates as expanded JSON on a rerun; the report
builder reads either form.

Individual checks:
    python3 -B original/phase2_8_row_factored_epistemic_gate.py
    python3 -B corrected/test_phase28.py
    python3 -B review/math_check.py
    python3 -B review/test_benchmark.py
    python3 -B review/audit_original.py
    python3 -B review/benchmark_phase28.py
    python3 -B review/build_report.py

Recorded execution: 64 passing unittest methods (3 original, 32 math/source,
21 corrected-engine, 8 harness); 48 exact original evaluator aggregate matches;
100+100 original seeds; 30 validation +100 primary +100 follow-up corrected seeds
across eight scenarios; 3,200 complete forecast-array matches to actual Phase 2.5
and its no-memory variant. One original test asserts performance on fixed seeds;
corrected invariant tests do not require the proposed model to win.

Main finding: pooled novel Brier/BCE improve over no-memory, chiefly from regime
E, while white-noise losses remain worse and follow-up D losses also worsen.
Direct row-write protection holds, but shared surprise and stale row buffers
persist. Surprise is (Y-p)(1-2p), not squared calibration error. Overall Brier/BCE
remain worse than actual Phase 2.5/2.6 on mixed_9. CUSUM wins overall Brier while
Phase 2.8 wins overall BCE against CUSUM; the scoring rule matters.

No phenomenological or physical MQGT-SCF conclusion follows from these functional
numerical experiments. All changes are additive; frozen artifacts are untouched.

Files:
original/   Exact extracted source/provenance. Private download also preserves
            uploaded text; the public repository excludes that raw conversation.
corrected/  Corrected engine, explicit ablations and deterministic tests.
reference/  Unchanged pinned prior engines and baseline classes.
review/     Mathematical audit, benchmark, harness tests, notes and source diff.
results/    Actual logs, per-seed scores, scoped event diagnostics, retrieval
            traces, validation manifest and reproduction/integrity records.

SHA256SUMS describes the delivered snapshot. Reruns regenerate results and may
change timing/environment log hashes. No architecture parameters were retuned
using corrected test outcomes. Intervals are exploratory, pointwise Monte Carlo
intervals; no multiple-comparison adjustment or equivalence margin was specified.
