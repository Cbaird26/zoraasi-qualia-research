PHASE 2.6 EXECUTED VERIFICATION — October 15, 2026

Start with VERIFICATION_REPORT.html or VERIFICATION_REPORT.txt.

Reproduce the complete audit (Python 3.10+, standard library only):
    python3 -B run_all.py

The original script is expected to exit 1 with its preserved SyntaxError.
A single-character repair is tested separately from the corrected engine.
All other execution/check stages must exit 0. Performance improvements are
measured results, not required software invariants.

Files:
  original/        Byte-exact supplied Python fence and provenance hashes.
  minimal_repair/  Only the invalid function-name space removed; no algorithm edit.
  corrected/       Corrected engine module and 20 invariant tests.
  reference/       Actual Phase 2.5 source pinned to commit 7be77aa, unchanged.
  review/          Benchmark harness, original audit, mathematics, harness tests,
                   implementation notes, source diffs and report renderer.
  results/         Executed logs, raw seed/block scores, event counts and intervals.
  run_all.py       Reproduction runner, including source-integrity verification.
  SHA256SUMS.txt   Delivery checksums; excludes itself.

Benchmark only:
    python3 -B review/benchmark_phase26.py --output results/corrected_benchmark.json

The corrected engine is an importable module; the separate benchmark harness
provides the executable comparison. Defaults: 30 validation, 100 primary and
100 follow-up seeds across seven scenarios. It saves its selection manifest
before testing and checks implementation hashes again after execution.

The private download retains the supplied pasted text. The public copy omits
the raw conversation and caches. Its largest result is stored losslessly as
results/corrected_benchmark.json.gz; the report renderer reads it directly.
Rerunning the suite regenerates the expanded JSON. All numerical evidence is
preserved. No external packages, network, GPU or API key are required.

Publication continues as an additive folder in PR #1:
https://github.com/Cbaird26/zoraasi-qualia-research/pull/1
Previous Phase 2.3–2.5 artifacts and the frozen public baseline are unchanged.
Functional prediction and internal numerical states only; no phenomenological
or physical-theory confirmation is claimed.
