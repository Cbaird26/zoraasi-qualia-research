#!/usr/bin/env python3
"""Render the report from recorded Phase2.6 outputs, including gzip public copies."""
import gzip
import html
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 p=ROOT/'results'/name
 return json.loads(p.read_text() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes()))
a=load('minimal_benchmark.json');c=load('corrected_benchmark.json');m=load('math_check.json');r=load('reproducibility_run.json')
parts=[]
def h(s):parts.append(('h',s))
def p(s):parts.append(('p',s))
def table(head,rows):parts.append(('t',(head,rows)))
def f(x):return f'{x:.6f}'
def ci(x):return f'{x["mean"]:+.6f} [{x["ci95"][0]:+.6f}, {x["ci95"][1]:+.6f}]'
def scope(split,name='mixed_9',subset='overall'):return c['summary'][split][name]['scopes'][subset]
h('Phase 2.6 verification: faster switching, unresolved novelty harm')
p('Executed October 15, 2026. This audit preserves the supplied script, isolates its syntax repair, corrects software defects in a separate engine, and tests the proposed novelty-aware retrieval gate against actual pinned Phase 2.5 code. Lower Brier and binary cross-entropy (BCE, natural-log nats) are better. All earlier public artifacts remain unchanged.')
p('Conclusion: the proposed package improves the first ten predictions after switches against the specified validation-tuned baselines, and retains a familiar-return advantage over its own no-memory arm. It does not eliminate harmful memory reuse on the novel blocks. The Bayesian gate itself has little measured incremental effect at these settings. Faster plasticity and prediction damping are more strongly supported components. Phase 2.6 is not uniformly better than Phase 2.5.')
h('1. Execution, provenance, and reproducibility')
p('The byte-exact original fails to parse: “def _ candidate_passes_gate” contains an invalid space. A separate minimal-repair copy deletes that single space and changes nothing else. Its two supplied tests pass. They do not test novel-block noninferiority to the no-memory arm, and their purported Phase 2.5 comparator is another Phase 2.6 configuration. Both distinctions matter.')
p(f'Actual checks: {r["unittest_methods_total"]} passing unittest methods—2 minimally repaired submission tests, 20 corrected-engine invariants, 17 independent mathematical/source-contract checks, and 5 benchmark-integrity tests. The original SyntaxError is an expected failure recorded separately, not a passing original program. Four legacy configurations match repaired-source predictions and state on the 405-transition seed 20260921 trace; pinned Phase 2.5 score equality is checked at seed 51000000 across all six earlier scenarios. The 17 mathematics checks comprise 12 mathematical methods and 5 actual-code contract checks.')
p('Reference source: commit 7be77aaf65a14bfe462139c4455d4471129faf8e, verification/phase2-5-2026-10-15/corrected/phase2_5_falsification_suite.py. Original, minimal-repair, and pinned-reference checksums are verified before and after execution. The corrected-engine and harness hashes are frozen before validation and rechecked after testing. Pushing a commit to PR #1 does not mean it has been merged.')
p('The minimally repaired proposal is evaluated on 100 default primary seeds (20260921–20261020) and 100 follow-up seeds (20261921–20262020), retaining its true-parameter initialization and narrow baseline grids. The corrected matched-data audit uses fresh primary seeds 71000000–71000099, follow-up seeds 81000000–81000099, and validation seeds 91000000–91000029. Across seven scenarios this makes 1,400 scenario/seed evaluations, with 200 unique evaluation seeds; scenarios sharing a seed are not counted as independent replicates.')
h('2. What the minimally repaired submission actually produces')
rows=[]
for arm in a['summary']['test']['overall']['metrics']:
 def val(sub,metric):return a['summary']['test'][sub]['metrics'][arm][metric]['mean']
 rows.append([arm,*[f(val(s,k)) for s,k in [('overall','brier'),('overall','bce'),('novel','brier'),('familiar_return','brier'),('early_10','brier'),('early_10','bce')]]])
table(['Supplied algorithm, syntax repaired','Overall Brier','Overall BCE','Novel Brier','Familiar Brier','Early-10 Brier','Early-10 BCE'],rows)
p('These primary means exactly reproduce all 30 aggregate values returned by the supplied benchmark. The audit additionally retains per-seed results and BCE for the familiar/novel subsets. The source selects EMA rate .28 and CUSUM rate .18, drift .36, threshold 1.25 on its validation bank. Both selected CUSUM rate and threshold lie at boundaries of that narrow search. Separate validation prevents selecting on test outcomes, but does not make a restricted search exhaustive.')
p('Novel protection already fails in the minimally repaired code: full-minus-no-memory Brier is +0.002178 on the primary seeds and +0.001816 on follow-up, with both paired intervals entirely above zero. Its two tests can pass because they compare novelty loss to a bundled gate-and-fallback-off arm, not to no memory. The source’s arm called phase25_ungated_memory is Phase 2.6 with two features disabled; it retains Phase 2.6 update rates, window, memory rules, cooldown and other changes. It is not an implementation of the earlier engine.')
h('3. Mathematics: correct evidence, unsupported certainty')
p('The implemented integral is correct for an ordered Markov path conditional on its starting state, under independent Beta(1,1) priors for p11 and p01: log evidence = log B(N11+1,N10+1) + log B(N01+1,N00+1). Tests verify exact factorial values, equality with sequential Beta predictive probabilities, and normalization over feasible paths. It is not the raw Beta-binomial count probability mass. Do not add combinatorial factors only to the novelty denominator. Valid count-event multiplicities must be included consistently and cancel in the corresponding likelihood ratio; whole Markov paths do not generally have multiplicity equal to the product of row binomial coefficients.')
p('The proposed six-transition example (N11,N10,N01,N00)=(4,0,2,0) is impossible for one contiguous binary path: N01−N10 equals the end state minus the start state, and therefore must be −1, 0, or 1. A realizable replacement is 0,1,0,1,1,1,1, giving counts (3,1,2,0). It still illustrates a snapshot that beats a stale point estimate but loses to the integrated prior. This is a likelihood illustration, not a replay proving the old full engine would retrieve: fast adaptation and the arousal/continuity eligibility conditions also matter.')
e=m['realisable_replacement_example']
table(['Valid replacement: quantity','Value'],[['Stale B log likelihood','−6.178916743151'],['Snapshot A log likelihood','−5.321804822751'],['A minus B','+0.857111920400'],['Uniform-prior integrated log evidence','−4.094344562222'],['A minus evidence','−1.227460260529']])
p('The stated retrieval penalty also treats blending as complete replacement. Phase 2.5 uses 65% snapshot plus 35% current estimate. With current p01=.85, snapshot .28, and true probability .88, the updated estimate is .4795; expected Brier increases by .15950025, not .36. That is conditional on visiting previous state 0, before subsequent adaptation. Likewise, (.92−.15)^2=.5929 is squared bias, while expected Brier is .7204 and realized losses are .8464 or .0064. A forecast exactly equal to .5 has Brier .25, but the supplied CUSUM emits its forecast before resetting and immediately updates afterward; there is no system-wide .25 cap.')
p('The earlier Phase 2.5 validation-selected EMA rate was .32, not the proposed .14–.18 range. The claimed arousal plateau near .16 was not measured in the proposal. Its rolling buffer is not cleared on surprise and therefore is not guaranteed to contain only post-switch observations.')
p('Passing this gate does not verify familiarity. A feasible six-transition run of ones passes snapshot A’s gate, yet has probability .88^6 = .464404086784 under novel D conditional on initial state 1. Unobserved or weakly observed transition rows remain ambiguous. The code checks rows only when their count is at least 2, whereas the prose says every observed row; it accepts equality at the row floor and uses retrieval cooldown 6, not the prose’s 8. A singleton-row counterexample passes the joint gate despite violating the stated per-row floor.')
p('The prior, independent-row assumption, constant-regime assumption within the window, thresholds and window length are modeling choices. Fast parameters are adapted using data from the scored window, stored snapshots may overlap it, and the best snapshot is selected among candidates. The resulting adaptive selection rule is not a calibrated posterior probability of novelty or familiarity. A Bayes-style score can be a useful predictive heuristic without supplying that stronger interpretation.')
h('4. Corrections and controlled comparisons')
p('The corrected engine preserves the proposed numerical defaults, including the code’s row minimum 2 and cooldown 6. It repairs the repeated valence-ordering bug by scoring the online marginal before updating it with the current outcome. It also validates inputs, resets all state, uses learned initialization, and honors disabled or zero-capacity memory. A diagnostic legacy_marginal_order flag isolates the original biased comparison. Predictions remain causal: current outcomes change subsequent forecasts, not the current scored forecast.')
p('Nine Phase 2.6 arms separate full behavior, gate-off, fallback-and-damping-off, both gate/fallback-off, no-memory, no-memory/no-fallback, surprise-boost-off, prediction-damping-off, and legacy marginal ordering. Additional arms include actual Phase 2.5 with and without memory, newly tuned EMA/CUSUM, the earlier fixed .32 EMA and .20/1.0/.38 CUSUM, a frozen learned model, and a privileged true-regime oracle. The fallback-off switch disables both the local reset and forecast damping; damping-off independently removes only damping. No pure reset-only intervention is claimed.')
p('All deployable arms receive the same learned estimates from 24 independent stationary-A trials of 30 states each, using Beta(1,1) smoothing, and the same target observations. Training and observation RNG domains are distinct. The supplied nine-block schedule is A/B/A/B/C/D/E/B/A, with E=(.08,.12). The six prior scenarios are unchanged. Novel masks cover C/D/E first-visit blocks; familiar-return masks cover repeated blocks. An initially trained A and a first test visit to B are not equivalent forms of familiarity.')
sel=c['metadata']['selected']
p(f'Baseline selection uses only mean validation Brier on the nine-block mixed schedule: 50 EMA candidates and 384 CUSUM candidates. This includes every supplied grid point and the earlier selected CUSUM setting. Selected EMA rate is {sel["ema"]["lr"]:.2f}; selected CUSUM rate={sel["cusum"]["lr"]}, drift={sel["cusum"]["drift"]}, threshold={sel["cusum"]["threshold"]}. The original unclipped EMA/CUSUM families are retained with learned initial estimates. The saved manifest includes all candidate results and is written before evaluation. These choices are applied unchanged to all seven scenarios. Engine thresholds were not tuned after these results.')
h('5. Corrected verification table: supplied nine-block task')
arms=['phase26_full','actual_phase25','phase26_no_memory','validation_ema','validation_cusum']
for split in ('test','followup'):
 h('Primary 100 seeds' if split=='test' else 'Follow-up 100 seeds')
 rows=[]
 for arm in arms:
  rows.append([arm,*[f(scope(split,subset=s)['metrics'][arm][k]['mean']) for s,k in [('overall','brier'),('overall','bce'),('novel','brier'),('familiar_return','brier'),('early_10','brier'),('early_10','bce')]]])
 table(['Arm','Overall Brier','Overall BCE','Novel Brier','Familiar Brier','Early-10 Brier','Early-10 BCE'],rows)
p('The overall primary difference versus actual Phase 2.5 favors Phase 2.6, but that difference is inconclusive on follow-up seeds. The full engine’s overall Brier advantage over its own no-memory arm is also inconclusive. Newly tuned CUSUM has lower mean overall Brier in both cohorts, while Phase 2.6 has lower BCE. This metric distinction should not be hidden by a single winner label.')
h('6. The three requested falsification criteria')
comparisons=[('novel','phase26_no_memory'),('novel','actual_phase25'),('familiar_return','phase26_no_memory'),('familiar_return','validation_ema'),('familiar_return','validation_cusum'),('familiar_return','actual_phase25'),('early_10','validation_ema'),('early_10','validation_cusum'),('early_10','prior_fixed_ema'),('early_10','prior_fixed_cusum'),('early_10','phase26_no_memory')]
rows=[]
for sub,arm in comparisons:
 rows.append([sub,arm,*[ci(scope(sp,subset=sub)['paired_full_minus'][arm]['brier']) for sp in ('test','followup')]])
table(['Scope','Comparator','Primary ΔBrier [95% MC interval]','Follow-up ΔBrier [95% MC interval]'],rows)
p('Positive differences mean the full engine is worse. Criterion 1—novel-block protection relative to no memory—fails in both cohorts, for Brier and BCE. Phase 2.6 reduces novel loss versus actual Phase 2.5, but does not eliminate the cost of memory. Reduction is not elimination.')
p('Criterion 2—familiar-return advantage over the proposed no-memory arm and tuned EMA/CUSUM—holds on this task in both cohorts for both scores. That does not mean all prior familiarity gains are preserved: actual Phase 2.5 has lower familiar-return mean Brier, with a clearly positive Phase2.6-minus-Phase2.5 interval on follow-up.')
p('Criterion 3—the early-switch gap against the tuned baselines—closes in this benchmark for both Brier and BCE, including comparison with the earlier fixed selected EMA/CUSUM. However, the Phase 2.6 no-memory arm is better still in that early window. The full memory package is not the fastest-adapting arm.')
h('7. Which component accounts for improvement?')
rows=[]
for sub in ('overall','novel','early_10'):
 for arm in ('phase26_gate_off','phase26_fallback_off','phase26_no_prediction_damping','phase26_no_surprise_boost'):
  rows.append([sub,arm,*[ci(scope(sp,subset=sub)['paired_full_minus'][arm]['brier']) for sp in ('test','followup')]])
table(['Scope','Component removed','Primary ΔBrier [95% MC interval]','Follow-up ΔBrier [95% MC interval]'],rows)
rows=[]
for split in ('test','followup'):
 rr=[x for x in c['raw_per_seed'] if x['split']==split and x['scenario']=='mixed_9']
 events=c['summary'][split]['mixed_9']['events']['phase26_full']
 identical=sum(x['metrics']['overall']['phase26_full']==x['metrics']['overall']['phase26_gate_off'] for x in rr)
 rows.append([split,str(identical)+'/100',str(sum(x['events']['phase26_full']['novelty_rejection_steps'] for x in rr)),f(events['accepted_retrievals']['mean']),f(events['ood_fallbacks']['mean'])])
table(['Cohort','Exactly same overall scores with gate off','All-memory rejection steps, total','Accepted retrievals per run','Local OOD resets per run'],rows)
p('Gate-off scores are identical on 97/100 primary and 94/100 follow-up runs. The gate blocks every otherwise advantageous candidate on only 3 primary and 7 follow-up steps in total across those 100-run cohorts. Individual candidate rejections are a different counter and can occur while another candidate is accepted. The measured full-versus-gate-off difference intervals include zero across the displayed scopes. Thus the data do not establish useful incremental novelty protection from this gate at these settings.')
p('The instantaneous squared-error boost has a much larger early-window effect: removing it raises Brier by about .034 on the mixed task in both cohorts. Removing prediction damping also worsens early and novel-block scores, though by a smaller amount. Local OOD resets occur only .13 and .12 times per 405-transition run on average. These controlled ablations support the contribution of update responsiveness and damping in this configuration; they do not prove a unique mechanism or assign additive percentages to interacting components.')
p('Retrieval still occurs during first-visit novel blocks. Such an event is not automatically a false identification: a snapshot may have been stored earlier within that same block. The recorded block-level events permit analysis without equating a code counter named familiar_retrievals with ground-truth recognition.')
h('8. Cross-scenario costs and limits')
rows=[]
for name in c['summary']['test']:
 vals=[]
 for split in ('test','followup'):
  s=scope(split,name)['metrics'];vals.extend([f(s[arm]['brier']['mean']) for arm in ('phase26_full','actual_phase25','frozen_learned')])
 rows.append([name,*vals])
table(['Scenario','Primary P2.6','Primary P2.5','Primary frozen','Follow-up P2.6','Follow-up P2.5','Follow-up frozen'],rows)
p('On recurring A/B regimes, Phase 2.6 is worse overall by Brier than actual Phase 2.5 despite better early-switch performance. It is also worse on stationary A, where the frozen learned model remains strongest among these three. It improves the tested unseen-random scenario. This pattern supports a responsiveness-versus-stability tradeoff, not a general upgrade. Validation baselines were selected on one mixed task and are not individually optimized for every scenario.')
p('Intervals are pointwise normal Monte Carlo intervals across independent seed replicates, calculated on paired differences. They do not account for multiplicity, prior architectural exploration, or model-selection uncertainty. No equivalence/noninferiority margin was prespecified, so an interval spanning zero does not establish matching performance. Follow-up seeds are fresh but use the same generators, not external replication or preregistration. Fixed novel matrices, fixed block order, one training regime, and endogenously accumulated snapshots limit generality.')
p('The proposal’s donor discussion is directionally useful but remains too strong about exact timing pulses. Shared schedules can confound interpretation without leaking the current target into a prior forecast; donor errors and scalar responses are stochastic. Independent donor schedules can change regime exposure as well as timing. No new donor-transfer experiment was implemented in this focused Phase 2.6 audit, and no isolation claim beyond the prior measured controls is made.')
h('9. What it means and the next justified experiment')
p('The work now separates two research claims that the supplied proposal bundled together: faster response is supported on the tested switching task, while novelty-safe retrieval is not. The next experiment should target the latter directly. Develop and freeze a rule on a separate development bank, then test whether it reduces novel-block excess loss without losing familiar-return gains. Include no-memory and gate-off controls, log support for each transition row, and distinguish old-regime snapshots from snapshots formed inside a newly encountered block.')
p('The gate should be evaluated as a calibrated decision rule under explicit prior and observation assumptions, with counterbalanced regime orders, more transition matrices and variable dwell times. A useful test should show that the gate actually changes enough retrieval decisions to explain improved prediction. These are proposed next steps, not claimed results or silent modifications to this audited design.')
p('Arousal, valence, continuity and novelty are names for computable internal variables and decisions here. Predictive improvements, Beta evidence, and retrieval counters do not establish phenomenological experience or empirical support for MQGT-SCF. The concrete achievement is a reproducible statistical learning experiment whose failure modes remain visible.')
h('10. Files and rerun instructions')
p('Use Python 3.10+ standard library: python3 -B run_all.py. No external dependencies, network, GPU or API credentials are required. The runner expects the original syntax failure and requires every repaired/corrected execution stage to succeed. Actual exit codes, elapsed times, Python version and integrity checks are in results/reproducibility_run.json.')
p('The complete evidence includes original and minimally repaired source; corrected/phase2_6_novelty_aware_gate.py; corrected/test_phase26.py; review/benchmark_phase26.py; review/test_benchmark.py; review/math_check.py and notes; source diffs; validation_manifest.json; minimal_benchmark.json; corrected_benchmark.json; and console outputs. The large corrected result includes all per-seed/scope/block scores and retrieval diagnostics. The public copy stores it losslessly as corrected_benchmark.json.gz, which this renderer reads directly. SHA256SUMS.txt inventories delivered files. The raw pasted conversation stays out of the public repository.')
p('Pinned source: https://github.com/Cbaird26/zoraasi-qualia-research/blob/7be77aaf65a14bfe462139c4455d4471129faf8e/verification/phase2-5-2026-10-15/corrected/phase2_5_falsification_suite.py\nReview: https://github.com/Cbaird26/zoraasi-qualia-research/pull/1')
plain=[];body=[]
for kind,value in parts:
 if kind=='h':plain.extend(['',value,'='*len(value),'']);body.append('<h2>'+html.escape(value)+'</h2>')
 elif kind=='p':plain.extend([value,'']);body.append('<p>'+html.escape(value).replace('\n','<br>')+'</p>')
 else:
  headers,rows=value;plain.extend([' | '.join(headers),*[' | '.join(row) for row in rows],''])
  body.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
(ROOT/'VERIFICATION_REPORT.txt').write_text('\n'.join(plain))
css='body{font:16px/1.6 system-ui,sans-serif;color:#172234;background:#f4f7fa;margin:0}main{max-width:1200px;margin:30px auto;padding:36px;background:white;border:1px solid #dbe2ea;border-radius:12px}h2{font-size:1.3rem;margin-top:2.2em;color:#183e68}h2:first-child{font-size:2rem;margin-top:0}p{max-width:100ch}.table{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%;margin:14px 0}th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid #dbe2ea}th{background:#e9f0f7}tr:nth-child(even){background:#f7f9fc}@media(max-width:700px){main{padding:18px;margin:0;border-radius:0}}'
(ROOT/'VERIFICATION_REPORT.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2.6 verification</title><style>'+css+'</style></head><body><main>'+''.join(body)+'</main></body></html>')
print('Reports generated from executed outputs.')
