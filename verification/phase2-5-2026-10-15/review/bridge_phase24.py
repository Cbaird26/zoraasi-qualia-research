#!/usr/bin/env python3
"""Compare pinned Phase 2.4 with Phase 2.5 on identical learned fits and streams."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

old = module('pinned_phase24', ROOT/'reference/phase24_engine.py')
new = module('corrected_phase25', ROOT/'corrected/phase2_5_falsification_suite.py')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark', type=Path, default=ROOT/'results/corrected_benchmark.json')
    ap.add_argument('--output', type=Path, default=ROOT/'results/bridge_phase24.json')
    args = ap.parse_args()
    benchmark = json.loads(args.benchmark.read_text() if args.benchmark.exists() else gzip.decompress(Path(str(args.benchmark)+'.gz').read_bytes()))
    records = []
    for reference in benchmark['raw_per_seed']:
        seed, scenario = reference['seed'], reference['scenario']
        schedule, q0 = new.make_scenario(scenario, seed)
        stream = new.generate_stream(schedule, seed, q0=q0)
        fit = new.generate_training(seed)
        models = {'phase24_full': old.SyntheticQualiaEngineV1(),
                  'phase24_no_self': old.SyntheticQualiaEngineV1(enable_self_model=False),
                  'phase25_legacy_order': new.RefinedQualiaEngineV1_1(legacy_marginal_order=True, rng_seed=seed),
                  'phase25_corrected': new.RefinedQualiaEngineV1_1(rng_seed=seed)}
        for name, model in models.items():
            if name.startswith('phase24'):
                model.initialize_from_estimates(fit.p11, fit.p01, fit.marginal)
            else:
                model.initialize(fit.p11, fit.p01, fit.marginal)
        paths = {name: [model.step_update(stream[t-1], stream[t]) for t in range(1,len(stream))]
                 for name, model in models.items()}
        scores = {name: new.score(stream[1:], path) for name, path in paths.items()}
        assert scores['phase25_corrected'] == reference['metrics']['overall']['full_engine']
        assert scores['phase25_legacy_order'] == reference['metrics']['overall']['legacy_marginal_order']
        records.append({'seed': seed, 'scenario': scenario, 'split': reference['split'], 'scores': scores})
    summary = {}
    pairs = [('phase24_no_self_minus_full','phase24_no_self','phase24_full'),
             ('phase25_corrected_minus_phase24_full','phase25_corrected','phase24_full'),
             ('phase25_corrected_minus_phase24_no_self','phase25_corrected','phase24_no_self'),
             ('phase25_corrected_minus_legacy_order','phase25_corrected','phase25_legacy_order')]
    for split in ('test','followup'):
        summary[split] = {}
        for scenario in new.SCENARIOS:
            rows = [r for r in records if r['split']==split and r['scenario']==scenario]
            if not rows: continue
            metrics = {arm: {m: new.mean_ci([r['scores'][arm][m] for r in rows]) for m in ('brier','bce')}
                       for arm in rows[0]['scores']}
            paired = {name: {m: new.mean_ci([r['scores'][a][m]-r['scores'][b][m] for r in rows])
                            for m in ('brier','bce')} for name,a,b in pairs}
            summary[split][scenario] = {'metrics':metrics,'paired_differences':paired}
            print(split, scenario)
            for arm, score in metrics.items():
                print(f'  {arm:26} Brier={score["brier"]["mean"]:.6f} BCE={score["bce"]["mean"]:.6f}')
            for name, score in paired.items():
                print(f'  {name}: {score["brier"]["mean"]:+.6f} {score["brier"]["normal_mc_ci95"]}')
    report = {'method':'All four models receive identical learned training estimates and test observations. Phase 2.4 algorithms are imported unmodified from pinned PR1 commit b25e8c2; B uses Phase2.5 parameters (.15,.85). No retuning. Phase2.5 metrics checked for exact equality against all corrected benchmark rows.',
              'limitations':['Removing Phase2.4 self-model is a bundled mechanistic intervention: its changed predictions also feed later error/scalar updates. It does not isolate a constant additive Brier bias.',
                             'Phase2.5 versus Phase2.4 no-self differs in several coefficients and retrieval/consolidation rules; it is a package comparison, not a single mechanism attribution.',
                             'Corrected versus legacy marginal order isolates the ordering fix on matched data; semantic validity does not require improved score.'],
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [ROOT/'reference/phase24_engine.py', ROOT/'corrected/phase2_5_falsification_suite.py']},
              'exact_benchmark_score_matches':2*len(records), 'summary':summary,'raw_per_seed':records}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__ == '__main__': main()
