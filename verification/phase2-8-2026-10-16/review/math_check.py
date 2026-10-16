#!/usr/bin/env python3
"""Independent Phase 2.8 mathematics and deterministic source-contract audit.

Run from the package root: python3 -B review/math_check.py --out results/math_check.json
Only imports preserved sources; no source is edited. Constructed states are not
claimed reachable from default initialization. Passing tests can refute claims.
"""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


p28 = load('math_p28_original', 'original/phase2_8_row_factored_epistemic_gate.py')
p25 = load('math_p25_reference', 'reference/phase25_suite.py')
p26 = load('math_p26_reference', 'reference/phase26_engine.py')
p27 = load('math_p27_reference', 'reference/phase27_engine.py')


def kl(p, q):
    return (p * math.log(p / q) if p else 0.) + ((1-p) * math.log((1-p)/(1-q)) if p < 1 else 0.)


def delta(y, p):
    return (y-p)**2 - p*(1-p)


def risk(p, q):
    return q*(1-p)**2 + (1-q)*p*p


def counts(path):
    n = [0, 0, 0, 0]  # N11, N10, N01, N00
    for prev, curr in zip(path, path[1:]):
        n[(1-prev)*2 + (1-curr)] += 1
    return tuple(n)


def probability(path, p11, p01, q0):
    pr = q0 if path[0] else 1-q0
    for prev, curr in zip(path, path[1:]):
        q = p11 if prev else p01
        pr *= q if curr else 1-q
    return pr


def ll(n, params):
    a, b, c, d = n
    p, q = params
    return a*math.log(p)+b*math.log1p(-p)+c*math.log(q)+d*math.log1p(-q)


def compatible(n, params, cutoff=.12):
    a, b, c, d = n
    return (a+b > 0 and c+d > 0 and
            kl((a+.5)/(a+b+1), params[0]) <= cutoff and
            kl((c+.5)/(c+d+1), params[1]) <= cutoff)


def enumerate_windows():
    out = {}
    for label, params, q0 in [('A_initial_one', (.92, .28), 1.),
                              ('A_stationary', (.92, .28), 7/9),
                              ('B_stationary', (.15, .85), .5)]:
        totals = dict(total=0., row0_absent=0., either_row_absent=0.,
                      balanced=0., balanced_row1_one_success=0.,
                      balanced_compatible=0., compatible=0.)
        paths = []
        for path in itertools.product((0, 1), repeat=7):
            n = counts(path); a, b, c, d = n
            pr = probability(path, *params, q0)
            totals['total'] += pr
            if c+d == 0: totals['row0_absent'] += pr
            if a+b == 0 or c+d == 0: totals['either_row_absent'] += pr
            good = compatible(n, params)
            if a+b == c+d == 3:
                totals['balanced'] += pr
                if a == 1: totals['balanced_row1_one_success'] += pr
                if good: totals['balanced_compatible'] += pr
            if good:
                totals['compatible'] += pr
                paths.append(dict(path=''.join(map(str, path)), counts=n, probability=pr))
        totals['compatible_given_balanced'] = totals['balanced_compatible']/totals['balanced']
        totals['row1_one_success_given_balanced'] = totals['balanced_row1_one_success']/totals['balanced']
        out[label] = dict(parameters=dict(p11=params[0], p01=params[1], q0=q0),
                          probabilities=totals, compatible_paths=paths)
    return out


def constructed_stale_buffer():
    e = p28.Phase28Engine()
    e.p11_fast = e.p01_fast = e.p11_slow = e.p01_slow = .5
    e.arousal = .65
    e.epistemic_surprise = .2
    e.row_buffers = [[], [1, 1, 1]]
    e.window6 = [(1, 1)]*5
    baseline = copy.deepcopy(e); baseline.mode = 'no_memory'
    forecast = e.step_update(1, 1)
    baseline.step_update(1, 1)
    q = .15  # new B row-1 truth; outcome 1 has positive probability
    return dict(scope='Constructed legal state, not asserted reachable from default initialization.',
                old_row_observations=3, new_row_observations=1, row_buffer=e.row_buffers[1],
                new_regime_row_probability=q, new_outcome_probability=q,
                forecast_before_update=forecast, post_base_fast=baseline.p11_fast,
                post_retrieval_fast=e.p11_fast, opposite_fast=e.p01_fast,
                opposite_fast_control=baseline.p01_fast, retrieved=e.row_retrievals,
                surprise_after=e.epistemic_surprise, arousal_after=e.arousal,
                continuity_after=e.continuity,
                empirical_mle=1.,
                empirical_squared_error_change=(e.p11_fast-1)**2-(baseline.p11_fast-1)**2,
                true_conditional_expected_brier_change=risk(e.p11_fast,q)-risk(baseline.p11_fast,q),
                alternate_calibrated_q_half_expected_brier_change=risk(e.p11_fast,.5)-risk(baseline.p11_fast,.5),
                cooldown_after=e.row_cooldown)


def constructed_cross_row_surprise():
    e = p28.Phase28Engine()
    e.p11_fast = e.p11_slow = .5
    e.p01_fast = e.p01_slow = .08
    e.arousal = .65
    e.row_buffers = [[], [1, 1, 1]]
    first_forecast = e.step_update(0, 1)
    surprise_from_row0 = e.epistemic_surprise
    intervention = copy.deepcopy(e)
    intervention.epistemic_surprise = 0.  # only this shared state is changed
    forecast = e.step_update(1, 1)
    intervention.step_update(1, 1)
    return dict(scope='Constructed state and one-variable causal intervention, not default-run frequency.',
                first_row0_forecast=first_forecast, surprise_from_row0=surprise_from_row0,
                next_row1_forecast=forecast, row1_local_delta=delta(1, forecast),
                retained_surprise=e.epistemic_surprise, reset_surprise=intervention.epistemic_surprise,
                retained_retrievals=e.row_retrievals, reset_retrievals=intervention.row_retrievals,
                retained_row1_fast=e.p11_fast, reset_row1_fast=intervention.p11_fast,
                retained_row0_fast=e.p01_fast, reset_row0_fast=intervention.p01_fast)


def noise_replay():
    seed = 20260921
    stream = p28.generate_scheduled_stream(p28.mixed_familiar_novel_schedule(), seed)
    models = [('phase25', p25.RefinedQualiaEngineV1_1()),
              ('phase26', p26.NoveltyAwareEngineV1_2()), ('phase27', p27.Phase27Engine())]
    result = {}
    for name, engine in models:
        engine.initialize(.92, .28, 7/9)
        trace = []
        key = 'retrievals' if name == 'phase25' else 'accepted_retrievals'
        for t, (prev, curr) in enumerate(zip(stream, stream[1:])):
            before = engine.stats.copy()
            pred = engine.step_update(prev, curr)
            if 180 <= t < 225:
                trace.append(dict(offset=t-180, previous=prev, current=curr,
                                  forecast=pred, arousal=engine.arousal, continuity=engine.continuity,
                                  eligible=engine.stats['retrieval_eligible']-before['retrieval_eligible'],
                                  accepted=engine.stats[key]-before[key]))
        result[name] = dict(eligible=sum(x['eligible'] for x in trace),
                            accepted=sum(x['accepted'] for x in trace), trace=trace)
    return dict(seed=seed, initialization='True A parameters, matching the supplied Phase 2.8 setup.',
                scope='One original-seed mixed schedule replay with pinned implementations; not an estimate across seeds.',
                stream_sha256=hashlib.sha256(bytes(stream)).hexdigest(), block4=result)


def original_vs_pinned():
    e = p28.Phase28Engine(mode='phase25_ungated')
    r = p25.RefinedQualiaEngineV1_1(); r.initialize(.92,.28,7/9)
    e.step_update(1,1);r.step_update(1,1)
    first = dict(original_valence=e.valence, pinned_valence=r.valence,
                 same_fast=(e.p11_fast,e.p01_fast)==(r.p11_fast,r.p01_fast))
    e = p28.Phase28Engine(mode='phase25_ungated')
    r = p25.RefinedQualiaEngineV1_1();r.initialize(.92,.28,7/9)
    stream=p28.generate_scheduled_stream(p28.mixed_familiar_novel_schedule(),20260921)
    a=[];b=[]
    for prev,curr in zip(stream,stream[1:]):
        a.append(e.step_update(prev,curr));b.append(r.step_update(prev,curr))
    unequal=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    return dict(first_update=first, seed=20260921, complete_forecasts_equal=not unequal,
                differing_forecasts=len(unequal), first_differing_forecast_index=unequal[0] if unequal else None,
                scope='One deterministic comparison, not a decomposition of aggregate performance differences.')


def constructed_calibrated_surprise_trigger():
    e=p28.Phase28Engine()
    e.arousal=.65
    e.row_buffers[1]=[1,1,1]
    forecast=e.step_update(1,0)
    return dict(scope='Constructed legal state, not claimed reachable from default initialization.',
                current_true_row_probability=.92,forecast=forecast,current_outcome=0,
                surprise=e.epistemic_surprise,arousal=e.arousal,continuity=e.continuity,
                failed_retrieval_checks=e.rejected_candidates,accepted=e.row_retrievals,
                interpretation='The calibrated issued forecast triggers a full retrieval check; all candidates are then rejected.')


def build_evidence():
    n=(4,0,1,1);candidate=(.92,.28);fast=(.60,.28);slow=(.70,.35)
    matches=[''.join(map(str,p)) for p in itertools.product((0,1),repeat=7) if counts(p)==n]
    gate=p27.Phase27Engine();gate.initialize(*fast,7/9)
    direct_pass=gate.candidate_passes_phase27_gate(p27.EpisodicSnapshot(*candidate,.5),*n,
                    ll(n,fast),ll(n,slow),slow_reference=slow)
    return dict(
        interpretation='Functional forecast-state audit. Statistical compatibility does not establish regime identity or phenomenology.',
        surprise=dict(identity='delta=(Y-p)(1-2p)', expectation='E_q delta=(q-p)(1-2p)',
                      variance='q(1-q)(1-2p)^2', ema_coefficient_in_prose=.25, ema_coefficient_in_code=.22,
                      calibrated_p92_delta_on_zero=delta(0,.92),
                      calibrated_p92_ema_after_one_zero=.22*delta(0,.92),
                      event_probability=.08, threshold=.08,
                      p_half_delta_both_outcomes=[delta(y,.5) for y in (0,1)],
                      negative_mean_despite_miscalibration=dict(p=.6,q=.92,mean=(.92-.6)*(1-2*.6)),
                      fixed_iid_calibrated_p92_stationary_ema_variance=.22/(2-.22)*.92*.08*(1-2*.92)**2,
                      caveat='Fixed-p EMA formula and one-outcome surprise event are not full-engine retrieval rates.'),
        contraction=dict(exact_squared_distance_change='w(c-f)[2(f-m)+w(c-f)]',
                         strict_contraction_iff='w(c-f)[2(f-m)+w(c-f)] < 0',
                         candidate_closeness='Sufficient for any fixed 0<w<=1, not necessary when w<1.',
                         not_necessary=dict(fast=.4,mle=.5,candidate=.65,weight=.65,blended=.5625,
                                            old_distance=.1,new_distance=.0625,candidate_distance=.15),
                         not_motion_away_from_slow=dict(fast=.4,mle=.75,candidate=.7,slow=.7,
                                                     blended=.595,old_slow_distance=.3,new_slow_distance=.105),
                         scope='Empirical-window contraction need not improve true-regime Brier or BCE.'),
        proposed_phase27_counterexample=dict(counts=n, realizable_paths=matches,
             distance=math.dist(candidate,slow), candidate_ll=ll(n,candidate),slow_ll=ll(n,slow),fast_ll=ll(n,fast),
             row1_kl=kl(.9,.92),row0_kl=kl(.5,.28),phase27_count_gate_passes=direct_pass,
             blended_p11=.65*.92+.35*.60,
             claim_conflict='All realizable paths end in 1, inconsistent with the asserted current 1->0 discordance. The supplied test substitutes row counts (3,1).'),
        quoted_kl=dict(KL_0_5_to_0_28=kl(.5,.28),KL_0_375_to_0_15=kl(.375,.15),KL_0_625_to_0_85=kl(.625,.85)),
        missing_row=dict(given_initial_one_row0_absent=.92**5,all_six_successes=.92**6,
                         explanation='Row exposure depends on X0..X5; the final outcome X6 is unrestricted.'),
        window_enumeration=enumerate_windows(),
        fixed_p_half=dict(arousal_target=.625,continuity_if_cosine_one=math.exp(-.85*.5),
                          from_arousal_point2_after_one=.7*.2+.3*.625,
                          caveat='Arousal equilibrium conditional on a frozen forecast; positive plasticity makes the forecast leave .5 after each outcome. Phase 2.6 uses different coefficients.'),
        row_gate_counts_at_n4={str(p):[k for k in range(5) if kl((k+.5)/5,p)<=.065] for p in (.92,.28,.15,.85)},
        stale_buffer=constructed_stale_buffer(),cross_row_surprise=constructed_cross_row_surprise(),
        calibrated_surprise_trigger=constructed_calibrated_surprise_trigger(),
        pinned_noise_replay=noise_replay(),original_vs_pinned=original_vs_pinned(),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                        [ROOT/'original/phase2_8_row_factored_epistemic_gate.py',ROOT/'reference/phase25_suite.py',
                         ROOT/'reference/phase26_engine.py',ROOT/'reference/phase27_engine.py']})


EVIDENCE=build_evidence()


class MathematicalChecks(unittest.TestCase):
    def test_surprise_identity(self):
        for p in (.01,.15,.28,.5,.6,.92,.99):
            for y in (0,1):self.assertAlmostEqual(delta(y,p),(y-p)*(1-2*p))

    def test_surprise_expectation(self):
        for p in (.1,.4,.5,.6,.9):
            for q in (.02,.15,.5,.92,.98):
                self.assertAlmostEqual(q*delta(1,p)+(1-q)*delta(0,p),(q-p)*(1-2*p))

    def test_true_expected_brier_decomposition(self):
        for p,q in itertools.product((.1,.5,.9),repeat=2):
            self.assertAlmostEqual(risk(p,q),q*(1-q)+(p-q)**2)

    def test_blindness_at_half(self):
        for q in (.01,.15,.5,.92,.99):
            self.assertEqual(q*delta(1,.5)+(1-q)*delta(0,.5),0)

    def test_calibration_zero_mean_not_no_variance(self):
        p=.92
        self.assertAlmostEqual(p*delta(1,p)+(1-p)*delta(0,p),0)
        self.assertGreater(.22*delta(0,p),.08)

    def test_negative_mean_despite_miscalibration(self):
        self.assertAlmostEqual((.92-.6)*(1-2*.6),-.064)

    def test_surprise_variance(self):
        p,q=.28,.88;m=(q-p)*(1-2*p)
        self.assertAlmostEqual(q*(delta(1,p)-m)**2+(1-q)*(delta(0,p)-m)**2,q*(1-q)*(1-2*p)**2)

    def test_exact_blend_contraction_identity(self):
        for f,m,c,w in itertools.product((.1,.4,.8),(.2,.5,.9),(.15,.65,.95),(.1,.65,1.)):
            r=f+w*(c-f)
            self.assertAlmostEqual((r-m)**2-(f-m)**2,w*(c-f)*(2*(f-m)+w*(c-f)))

    def test_closeness_sufficient(self):
        for f,m,c,w in itertools.product((.1,.4,.8),(.2,.5,.9),(.15,.65,.95),(.1,.65,1.)):
            if abs(c-m)<abs(f-m):self.assertLess(abs(f+w*(c-f)-m),abs(f-m))

    def test_closeness_not_necessary(self):
        f,m,c,w=.4,.5,.65,.65
        self.assertGreater(abs(c-m),abs(f-m));self.assertLess(abs(f+w*(c-f)-m),abs(f-m))

    def test_path_flow_and_counterexample_last_outcome(self):
        for path in itertools.product((0,1),repeat=7):
            a,b,c,d=counts(path);self.assertEqual(c-b,path[-1]-path[0])
        self.assertEqual(EVIDENCE['proposed_phase27_counterexample']['realizable_paths'],['0011111'])

    def test_missing_row_exponent(self):
        v=EVIDENCE['window_enumeration']['A_initial_one']['probabilities']
        self.assertAlmostEqual(v['row0_absent'],.92**5)
        self.assertGreater(v['row0_absent'],.92**6)

    def test_markov_paths_normalize(self):
        for v in EVIDENCE['window_enumeration'].values():self.assertAlmostEqual(v['probabilities']['total'],1.)

    def test_B_balanced_not_binomial(self):
        v=EVIDENCE['window_enumeration']['B_stationary']['probabilities']
        self.assertAlmostEqual(v['balanced'],.660885625)
        self.assertAlmostEqual(v['row1_one_success_given_balanced'],.2671000064307647)
        self.assertNotAlmostEqual(v['row1_one_success_given_balanced'],3*.15*.85**2)

    def test_B_conditional_vs_joint_pass(self):
        v=EVIDENCE['window_enumeration']['B_stationary']['probabilities']
        self.assertAlmostEqual(v['balanced_compatible'],.85**6)
        self.assertAlmostEqual(v['compatible_given_balanced'],.5706728991495313)
        self.assertAlmostEqual(v['compatible'],.57681690625)

    def test_exact_quoted_numbers(self):
        self.assertAlmostEqual(kl(.5,.28),.10758769083251637)
        self.assertAlmostEqual(kl(.375,.15),.15143108711033282)
        v=EVIDENCE['proposed_phase27_counterexample']
        self.assertAlmostEqual(v['candidate_ll'],-1.9349961785411274)
        self.assertAlmostEqual(v['blended_p11'],.808)


class SourceContractChecks(unittest.TestCase):
    def test_p27_count_gate_passes_but_not_discordant_path(self):
        self.assertTrue(EVIDENCE['proposed_phase27_counterexample']['phase27_count_gate_passes'])

    def test_claimed_necessary_condition_rejects_valid_contraction(self):
        self.assertFalse(p28.Phase28Engine.passes_directional_contraction(.65,.4,.1,2,2,1))

    def test_latest_outcome_direction_not_required_for_contraction(self):
        self.assertFalse(p28.Phase28Engine.passes_directional_contraction(.9,.6,.2,4,1,0))
        self.assertLess(abs(.65*.9+.35*.6-.8),abs(.6-.8))

    def test_passed_heuristic_can_move_toward_slow(self):
        self.assertTrue(p28.Phase28Engine.passes_directional_contraction(.7,.4,.7,3,1,1))
        self.assertLess(abs(.65*.7+.35*.4-.7),abs(.4-.7))

    def test_stale_buffer_first_new_transition_can_retrieve(self):
        v=EVIDENCE['stale_buffer'];self.assertEqual(v['retrieved'],1)
        self.assertEqual(v['old_row_observations'],3);self.assertEqual(v['new_row_observations'],1)

    def test_empirical_contraction_can_harm_true_risk(self):
        v=EVIDENCE['stale_buffer']
        self.assertLess(v['empirical_squared_error_change'],0)
        self.assertGreater(v['true_conditional_expected_brier_change'],0)

    def test_direct_opposite_fast_row_protected(self):
        v=EVIDENCE['stale_buffer'];self.assertEqual(v['opposite_fast'],v['opposite_fast_control'])

    def test_global_surprise_couples_rows(self):
        v=EVIDENCE['cross_row_surprise']
        self.assertEqual(v['row1_local_delta'],0)
        self.assertEqual(v['retained_retrievals'],1);self.assertEqual(v['reset_retrievals'],0)
        self.assertEqual(v['retained_row0_fast'],v['reset_row0_fast'])

    def test_row_buffer_survives_other_row_updates(self):
        e=p28.Phase28Engine(mode='no_memory');e.row_buffers[1]=[0,1,1,0]
        for _ in range(20):e.step_update(0,0)
        self.assertEqual(e.row_buffers[1],[0,1,1,0])

    def test_white_noise_half_not_dynamic_fixed_point(self):
        e=p28.Phase28Engine(mode='no_memory')
        e.p11_fast=e.p01_fast=e.p11_slow=e.p01_slow=.5;e.arousal=.625
        e.step_update(1,1)
        self.assertNotEqual(e.p11_fast,.5);self.assertEqual(e.epistemic_surprise,0)

    def test_positive_carried_surprise_not_reset_by_half(self):
        e=p28.Phase28Engine(mode='no_memory');e.p11_fast=.5;e.epistemic_surprise=.2
        e.step_update(1,1)
        self.assertAlmostEqual(e.epistemic_surprise,.156)
        self.assertGreater(e.epistemic_surprise,.08)

    def test_calibrated_forecast_can_trigger_full_retrieval_check(self):
        v=EVIDENCE['calibrated_surprise_trigger']
        self.assertEqual(v['forecast'],v['current_true_row_probability'])
        self.assertEqual(v['failed_retrieval_checks'],1);self.assertEqual(v['accepted'],0)

    def test_pinned_noise_gate_not_permanently_open(self):
        v=EVIDENCE['pinned_noise_replay']['block4']
        self.assertEqual([v[x]['eligible'] for x in ('phase25','phase26','phase27')],[18,17,19])

    def test_original_not_exact_pinned_phase25(self):
        v=EVIDENCE['original_vs_pinned']
        self.assertTrue(v['first_update']['same_fast'])
        self.assertNotEqual(v['first_update']['original_valence'],v['first_update']['pinned_valence'])
        self.assertFalse(v['complete_forecasts_equal'])

    def test_cooldown_is_global_step_not_active_row_count(self):
        e=p28.Phase28Engine(mode='no_memory');e.row_cooldown=[0,12]
        observed=[]
        for _ in range(12):e.step_update(0,0);observed.append(e.row_cooldown[1])
        self.assertEqual(observed,list(range(11,-1,-1)))

    def test_n4_kl_can_accept_or_reject_true_rows(self):
        self.assertEqual(EVIDENCE['row_gate_counts_at_n4'],{'0.92':[4],'0.28':[1],'0.15':[0],'0.85':[4]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results/math_check.json')
    args=parser.parse_args()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                             for c in (MathematicalChecks,SourceContractChecks))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    EVIDENCE['tests']=dict(run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                           mathematical_methods=unittest.defaultTestLoader.loadTestsFromTestCase(MathematicalChecks).countTestCases(),
                           source_contract_methods=unittest.defaultTestLoader.loadTestsFromTestCase(SourceContractChecks).countTestCases())
    EVIDENCE['status']='passed' if result.wasSuccessful() else 'failed'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(EVIDENCE,indent=2,sort_keys=True)+'\n')
    print(f'Wrote {args.out}; tests={result.testsRun}; successful={result.wasSuccessful()}')
    print('Important: passing counterexample tests refutes guarantees; it does not validate scientific hypotheses.')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
