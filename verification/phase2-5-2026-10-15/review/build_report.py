#!/usr/bin/env python3
"""Render readable reports from executed outputs; no additional experiments."""
import gzip
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(n):
    path=ROOT/'results'/n
    return json.loads(path.read_text() if path.exists() else gzip.decompress(Path(str(path)+'.gz').read_bytes()))
a=load('original_benchmark.json');c=load('corrected_benchmark.json');m=load('math_check.json');b=load('bridge_phase24.json')
sections=[]
def text(s):sections.append(('p',s))
def heading(s):sections.append(('h',s))
def table(headers,rows):sections.append(('t',(headers,rows)))
def f(x):return f'{x:.6f}'
def ci(x):return f'{x["mean"]:+.6f} [{x["normal_mc_ci95"][0]:+.6f}, {x["normal_mc_ci95"][1]:+.6f}]'
def scope(split,scenario,subset='overall'):return c['summary'][split][scenario]['scopes'][subset]

heading('Phase 2.5 verification and interpretation')
text('Executed October 15, 2026. Lower Brier and binary cross-entropy (BCE, natural-log nats) are better. This report audits the supplied Phase 2.5 proposal and compares it with the pinned Phase 2.4 implementation from PR #1. It does not modify the frozen Phase 2.3 baseline or the previous verification package.')
text('Finding: the revised engine improves overall prediction on the tested recurring-regime task and survives the two specified validation-tuned adaptive baselines. It does not establish faster adaptation than those baselines: its first ten predictions after switches have higher Brier loss. The corresponding BCE differences are inconclusive. Episodic memory helps familiar returns but harms the specified novel regimes. These are useful, narrower findings than a generally superior adaptation mechanism.')
heading('1. Provenance and executed checks')
text('The original Python fence was extracted byte-for-byte from the supplied attachment and executed unchanged. Its SHA256 is 14d7bae0e82bf79b0f94073ab18c7d0539fc42961a94020cce5e90f74a5dc756. The prior engine is pinned to commit b25e8c235df5078f17431b20ceaa914764dd06e1, path verification/2026-10-15/corrected/phase2_4_functional_state_engine.py, SHA256 eafca48ee34b912bcf05cf3fdb9e23bed817bbb1a9272d7db54d9525b7b83de2. Checksums are verified before and after the reproducibility run.')
text('Executed checks: 2 supplied default-seed performance assertions; 18 corrected software-invariant tests; 11 mathematical tests; 2 additional Phase 2.5 chronology checks; and 1 Phase 2.4 stored-continuity audit (256 paths, 1,792 paired transitions). All passed. The same-data bridge additionally matched 2,400 Phase 2.5 score dictionaries exactly. The 31 unittest methods are distinct from these additional audits and from the prior package’s 44 tests.')
text('The original proposal was evaluated over 200 seeds (20260921–20261120). The corrected suite uses 30 disjoint validation seeds (41000000–41000029), 100 primary seeds (51000000–51000099), and 100 follow-up seeds (61000000–61000099), across six scenarios: 1,200 scenario/seed evaluations. Scenarios share seed identifiers deliberately; they are not 1,200 independent seed replicates. No engine constants were tuned after these corrected results.')
heading('2. What the unchanged submission actually does')
rows=[]
for ex,arms in a['summary'].items():
 for arm,v in arms.items():
  one=a['single_seed'][ex][arm]
  rows.append([ex,arm,f(one['brier']),f(one['bce']),f(v['brier']['mean']),f(v['bce']['mean'])])
table(['Experiment','Arm','Default Brier','Default BCE','200-seed Brier','200-seed BCE'],rows)
text('The default engine score, 0.157108, is slightly worse than the hindsight-selected EMA, 0.156976. Averaging 200 seeds reverses that comparison. The submission explicitly calls this EMA an oracle; it chooses a separate rate using each test stream’s outcomes, so it is a hindsight diagnostic rather than a deployable validation-selected comparator. Its actual grid is 0.01–0.44, although the prose advertises 0.01–0.50. BCE was defined but never printed; the audit records it on exactly the predictions used by the original Brier calls, with no change to predictions.')
text('The assertion that memory helps revisited blocks fails on 13 of the 200 additional seeds. The endogenous-versus-offset assertion fails on none. Seed-specific model wins are not software invariants; the corrected tests check causality, reset isolation, formulas, bounds, RNG separation, controls, and score arithmetic without requiring a model to win.')
text('Other discrepancies: initialization uses the true A parameters rather than an estimated training fit; the advertised random-snapshot arm is absent; the offset schedule ends with consecutive B45 and B23 blocks, making a 68-step final dwell; and no independent stochastic donor dwell schedule was implemented. Regime B changes from Phase 2.4 (.18,.82) to (.15,.85), along with several algorithm coefficients. The old 0.185283 score and new 0.157108 score therefore cannot be interpreted as the effect of removing shrinkage alone.')
heading('3. Mathematical corrections')
text('For blend p = w f + (1-w)m and true conditional probability q, the extra expected Brier risk over f is 2(1-w)(f-q)(m-f) + (1-w)^2(m-f)^2. Only when f=q does this reduce to the squared bias term, and that term still includes (1-w)^2. Shrinkage toward 0.5 or a fast marginal does not guarantee improvement. For example, if f=q=.82, w=.5, replacing a .7778 prior with .5 increases the bias.')
text('The finite 29-transition calculation uses known p11=.92, p01=.28, q0=P(X0=1), stationary pi=7/9, and lambda=.64. Reset predicts using the trial’s own X0 at every step; deranged uses an independent trial’s preceding state. The exact expectations below exclude finite-training estimation noise. They are finite-horizon averages, not pointwise t-to-infinity limits.')
rows=[]
for q,row in m['initialization_table'].items():
 r,d=row['average']['reset'],row['average']['deranged']
 rows.append([q,f(r['brier']),f(d['brier']),f(r['bce_nats']),f(d['bce_nats'])])
table(['q0','Reset Brier','Deranged Brier','Reset BCE','Deranged BCE'],rows)
text('The reset-minus-deranged crossover is q0=0.696113688295 for Brier and q0=0.656197028823 for BCE. Below each threshold reset is worse; between the thresholds the metrics disagree. At stationarity the reset BCE is 0.687342669157. The deranged pointwise long-run Brier is 0.243634567901 for every q0; 0.2504 is neither that limit nor the correct q0=.5 finite-horizon mean. The independent derivation and tests are in review/math_check.py and review/math_notes.txt.')
heading('4. Causality, terminology, and implemented corrections')
text('A synchronous donor shares a regime schedule and therefore is not a timing-independent control. However, this does not establish current-target or future-target leakage. In the supplied Phase 2.5 code the recipient forecast p_t is computed before its current update, so donor scalars computed after observing the donor outcome at t change forecasts from t+1 onward, not p_t. Such an online donor requires contemporaneous donor observations to be available. The corrected suite explicitly issues all current forecasts before revealing any current outcome, and includes synchronous, circularly shifted, and independently timed donors.')
text('The proposal’s stronger donor explanation is not supported by the old code: transferred stored continuity in Phase 2.4 is overwritten before it is read, and varying only that stored value leaves prediction and updated state identical. Donor error and arousal are stochastic; they do not spike at exactly every change point. Shared timing remains a confound for interpreting donor controls, not proof of the proposed exact causal story.')
text('There is a separate real ordering bias in Phase 2.5: the marginal comparator is updated with y_t before its loss on y_t is computed. That gives the comparator an in-sample advantage in valence. The corrected engine compares two forecasts made before y_t, then updates the marginal. A legacy_marginal_order arm isolates this change; fixing the comparison does not imply better prediction on every task.')
text('The supplied named Page-Hinkley learner accumulates fixed-reference absolute error minus .38. It is reported here as an error-CUSUM comparator. For comparison, River’s published Page-Hinkley implementation tracks a running mean and centers updates on it. The corrected suite tunes the supplied fixed-reference family; it does not claim to implement or exhaust standard changepoint algorithms. Primary reference: https://github.com/online-ml/river/blob/main/river/drift/page_hinkley.py')
text('Additional corrections: complete state/RNG reset; input and probability validation; genuinely empty memory at zero capacity; no probability clipping update when learning rate is zero; learned Beta(1,1) transition fits from 24 independent 30-state training trials; BCE reporting; full .01–.50 EMA grid; selected-parameter manifest saved before evaluation; random eligible snapshot candidate with the same likelihood acceptance gate; explicit retrieval/store counters; and block-level, novel/revisited, initial-30, and early-10 scores. The original source is retained unchanged.')
heading('5. Corrected benchmark: honest validation selection')
sel=c['metadata']['selected_baselines']
text(f'Only mean Brier on the 30 validation recurrent_45 streams selects baselines. EMA: 50 candidate rates, selected {sel["ema_rate"]:.2f}. Error-CUSUM: 108 rate/threshold/reference combinations, selected rate={sel["cusum_rate"]}, threshold={sel["cusum_threshold"]}, reference={sel["cusum_reference"]}. The choices and every candidate’s per-seed scores are stored in results/corrected_validation_manifest.json before the test loops. BCE is secondary; the same Brier-selected models are evaluated on all scenarios. Training, recipient observations, donor observations, schedules, and random retrieval have distinct reproducible RNG domains.')
rows=[]
for arm,v in scope('test','recurrent_45')['metrics'].items():
 w=scope('followup','recurrent_45')['metrics'][arm]
 rows.append([arm,f(v['brier']['mean']),f(v['bce']['mean']),f(w['brier']['mean']),f(w['bce']['mean'])])
table(['Recurring 45-step regimes','Primary Brier','Primary BCE','Follow-up Brier','Follow-up BCE'],rows)
text('The true-regime oracle has privileged transition probabilities and is a diagnostic lower-risk reference, not a deployable competitor. The frozen learned Markov arm is useful for distinguishing beneficial adaptation from unnecessary tracking noise.')
rows=[]
for sc in c['summary']['test']:
 x=scope('test',sc)['metrics'];y=scope('followup',sc)['metrics']
 rows.append([sc,f(x['full_engine']['brier']['mean']),f(x['validation_tuned_ema']['brier']['mean']),f(x['validation_tuned_cusum']['brier']['mean']),f(x['frozen_learned_markov']['brier']['mean']),f(y['full_engine']['brier']['mean'])])
table(['Scenario','Full Brier','Tuned EMA','Tuned CUSUM','Frozen learned','Follow-up full'],rows)
text('The engine’s mean Brier is lower than both selected adaptive baselines in all six scenarios, but this does not establish universal superiority. Both baselines were tuned on recurring A/B regimes, not separately for each scenario. On stationary A the frozen learned model is substantially better than the adaptive engine. A stationary-appropriate comparator matters.')
heading('6. Falsifications that survive the corrections')
rows=[]
comparisons=[('recurrent_45','overall','validation_tuned_ema'),('recurrent_45','overall','validation_tuned_cusum'),('recurrent_45','early_10_after_change','validation_tuned_ema'),('recurrent_45','early_10_after_change','validation_tuned_cusum'),('recurrent_45','early_10_after_change','no_memory'),('familiar_novel','novel_C_D','no_memory'),('familiar_novel','novel_C_D','random_snapshot'),('familiar_novel','postnovel_revisited','no_memory'),('familiar_novel','postnovel_revisited','random_snapshot')]
for sc,sub,arm in comparisons:
 vals=[scope(sp,sc,sub)['paired_full_minus_comparator'][arm]['brier'] for sp in ('test','followup')]
 rows.append([sc+'/'+sub,arm,*map(ci,vals)])
table(['Comparison scope','Comparator','Primary full-minus-comparator Brier [95% MC interval]','Follow-up difference [95% MC interval]'],rows)
text('Positive differences mean the full engine is worse. Its lower overall recurring loss coexists with higher Brier loss in the first ten steps after changes than either tuned adaptive baseline. The early-window BCE difference intervals span zero for both baselines in both cohorts, so BCE does not establish the same conclusion. Memory improves those early steps relative to the same engine without memory, but that is a narrower comparison. Thus, “memory contributes here” is supported; “faster than tuned baselines” is contradicted by this window.')
text('The conjunction “helps familiar returns without harming novelty” fails. On the fixed C/D novel blocks, full memory is worse than no memory and random-candidate retrieval in both seed cohorts. Following novelty, it helps when A/B return. This is consistent with negative transfer from old snapshots, but it is not a fully isolated causal proof of novelty detection failure: C/D differ in transition structure and difficulty, the order is fixed, and A is also present in training.')
e=c['summary']['test']['recurrent_45']['events']
text(f'Control activity was measured rather than assumed. On recurring primary streams, random-snapshot retrieval is eligible {e["random_snapshot"]["retrieval_eligible"]["mean"]:.2f} times per seed; the random candidate happens to be the best {e["random_snapshot"]["candidate_is_best"]["mean"]:.2f} times; it accepts {e["random_snapshot"]["retrievals"]["mean"]:.2f} retrievals versus {e["full_engine"]["retrievals"]["mean"]:.2f} for the full engine. This random control changes candidate selection while preserving the likelihood gate; it is not unrestricted random retrieval or a matched-count intervention.')
text('Endogenous signals beat synchronous, offset, and independent donors on the recurring task. That establishes task-specific predictive value for alignment of these scalar updates under the stated interventions. It does not identify an affective mechanism, prove the scalars are necessary, or rule out a simpler error-based learning-rate schedule. On stationary A the synchronous and offset donor variants have lower Brier than the endogenous engine. Their BCE differences are small and inconclusive, further limiting any universal claim.')
heading('7. Phase 2.4 versus Phase 2.5 on identical data')
rows=[]
for sc,data in b['summary']['test'].items():
 d=data['metrics']
 rows.append([sc,*[f(d[k]['brier']['mean']) for k in ('phase24_full','phase24_no_self','phase25_legacy_order','phase25_corrected')]])
table(['Primary scenario','Phase 2.4 full','Phase 2.4 no self-model','Phase 2.5 legacy order','Phase 2.5 corrected'],rows)
text('This bridge imports the unmodified pinned Phase 2.4 engine and replays exactly the same learned fits and target paths as the corrected Phase 2.5 benchmark. It preserves each engine’s own update equations. Removing the Phase 2.4 self-model improves mean scores throughout this bridge. Phase 2.5 adds further improvement on recurring A/B tasks, but is worse than Phase 2.4 without the self-model on stationary A; the unseen-random task also shows no general extra benefit. Differences between Phase 2.5 and Phase 2.4 no-self bundle several update and memory-rule changes.')
text('The isolated marginal-order fix has small, mixed effects. On recurrent primary seeds it changes mean Brier from 0.160373 to 0.160388; on follow-up seeds it changes 0.161406 to 0.161350. Its purpose is a fair pre-observation comparison, not guaranteed score improvement. Original-versus-corrected file differences are supplied for review.')
heading('8. What this means, and the next falsifiable steps')
text('The positive result is an inspectable adaptive predictor with measurable memory tradeoffs. Simulated arousal, valence, and continuity are numerical state variables whose update rules can be ablated and challenged. Their names do not convert predictive gains into evidence of subjective feeling. No experiment here operationalizes phenomenology or distinguishes MQGT-SCF from competing physical theories; neither is validated by this benchmark.')
text('The next intervention should target inappropriate retrieval, rather than adding more named internal states: define a novelty or out-of-distribution rejection rule on a separate development set; freeze it; then test whether it reduces novel-regime excess loss without sacrificing familiar-return gains. Include memory retrieval count and matched-count random controls, and compare a plain transition-change detector with the same information budget.')
text('A stronger adaptation test should vary switch hazards and dwell lengths, use independently sampled and counterbalanced novel/familiar transition matrices, and report recovery curves as well as overall proper scores. A general-purpose performance claim also needs independently tuned stationary and switching baseline families, a no-memory engine with validation-selected rates, and uncertainty-aware model-selection comparators. These are proposed next experiments, not results claimed by this package.')
text('Statistical limits: intervals are pointwise normal Monte Carlo mean intervals across independent seed replicates, computed on paired within-seed score differences; blocks and transitions are not treated as independent samples. They do not include model-selection uncertainty or multiplicity correction. The follow-up uses disjoint seeds but the same author-designed generators and is not external replication or preregistration. One stationary regime, a fixed novel schedule, a small set of transition matrices, and tuning on one family limit generalization. The independent donor can differ in regime exposure as well as timing, so its disadvantage is not uniquely attributable to timing.')
heading('9. Reproduction and files')
text('Requirements: Python 3.10+ and its standard library; no external packages, network, GPU, or API keys. Run from this directory: python3 -B run_all.py. The runner verifies original/reference hashes, executes the original default run and 200-seed audit, mathematical checks, corrected invariant tests, validation plus 100+100 seeds across six scenarios, and the common-data bridge. It regenerates this report after the run. See results/reproducibility_run.json for observed environment, exit codes, and elapsed times.')
text('Primary files: corrected/phase2_5_falsification_suite.py; corrected/test_phase25.py; review/math_check.py; review/bridge_phase24.py; review/audit_original.py. Data: results/original_benchmark.json, results/corrected_validation_manifest.json, results/corrected_benchmark.json, results/bridge_phase24.json. The large corrected JSON (stored with lossless .gz compression in the public repository) retains per-seed scores, block metrics, donor schedules, and event counters. Actual console logs are alongside it. SHA256SUMS.txt inventories the delivery. The private download may include the pasted source text; public copies omit that raw conversation and still verify the extracted script hash.')
text('Pinned prior source: https://github.com/Cbaird26/zoraasi-qualia-research/blob/b25e8c235df5078f17431b20ceaa914764dd06e1/verification/2026-10-15/corrected/phase2_4_functional_state_engine.py\nReview: https://github.com/Cbaird26/zoraasi-qualia-research/pull/1')

plain=[];body=[]
for kind,value in sections:
 if kind=='h':
  plain.extend(['',value,'='*len(value),'']);body.append('<h2>'+html.escape(value)+'</h2>')
 elif kind=='p':
  plain.extend([value,'']);body.append('<p>'+html.escape(value).replace('\n','<br>')+'</p>')
 else:
  heads,rows=value
  plain.extend([' | '.join(heads),*[' | '.join(row) for row in rows],''])
  body.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
(ROOT/'VERIFICATION_REPORT.txt').write_text('\n'.join(plain))
style='body{font:16px/1.6 system-ui,sans-serif;color:#172234;background:#f5f7fa;margin:0}main{max-width:1180px;margin:30px auto;background:white;padding:36px;border:1px solid #dbe2ea;border-radius:12px}h2{font-size:1.35rem;margin-top:2.2em;color:#183e68}h2:first-child{font-size:2rem;margin-top:0}p{max-width:100ch}.table{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%;margin:14px 0}th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid #dbe2ea}th{background:#e9f0f7}tr:nth-child(even){background:#f7f9fc}@media(max-width:700px){main{padding:18px;margin:0;border-radius:0}}'
(ROOT/'VERIFICATION_REPORT.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2.5 verification</title><style>'+style+'</style></head><body><main>'+''.join(body)+'</main></body></html>')
print('Reports generated from executed results.')
