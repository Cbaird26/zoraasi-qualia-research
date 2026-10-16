PHASE 2.5 EXECUTED VERIFICATION — October 15, 2026

Start with VERIFICATION_REPORT.html or VERIFICATION_REPORT.txt.

Reproduce everything using Python 3.10+ standard library:
    python3 -B run_all.py

Observed complete computation time was approximately 40 seconds on the recorded
local environment; other computers may differ. No packages or network required.
The runner exits nonzero for an execution/check failure. Model performance is
reported as an experimental result and is not required to improve.

Contents:
  original/       Unmodified extracted script and provenance checksums.
  corrected/      Corrected suite and 18 invariant tests.
  reference/      Pinned prior Phase 2.4 engine from PR1, unchanged.
  review/         Independent math audit, original-score audit, same-data bridge,
                  implementation notes, source diff, and report renderer.
  results/        Actual logs, raw per-seed data, validation choices, intervals.
  run_all.py      Complete rerun with integrity checks.
  SHA256SUMS.txt  Delivered-file checksum inventory (excludes itself).

The private bundle retains the user-pasted source text. The public repository
omits that raw conversation and Python caches. Source-script and pinned-reference
checks still work without the raw conversation. The largest public result is
losslessly compressed as results/corrected_benchmark.json.gz. Report and bridge
readers accept that file directly; rerunning the suite regenerates the expanded
JSON. No data is omitted by compression.

The public follow-up is additive under verification/phase2-5-2026-10-15/ in PR1:
https://github.com/Cbaird26/zoraasi-qualia-research/pull/1
Prior Phase 2.3/2.4 files and frozen baseline are untouched.

Scientific scope: functional numerical-state simulation and prediction only;
no evidence of phenomenology or validation of MQGT-SCF is claimed.
