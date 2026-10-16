#!/usr/bin/env python3
"""Render the executed Phase 2.8 audit; accepts losslessly compressed result JSON."""
from pathlib import Path
import gzip
import html
import json
ROOT=Path(__file__).resolve().parents[1]
def read(name):
 p=ROOT/'results'/name
 return json.loads(p.read_text() if p.exists() else gzip.decompress(p.with_suffix(p.suffix+'.gz').read_bytes()))
d=read('corrected_benchmark.json');o=read('original_benchmark.json');m=read('math_check.json');r=read('reproducibility_run.json')
txt=[];ht=[]
def title(s):txt.extend([s,'='*len(s),'']);ht.append('<h1>'+html.escape(s)+'</h1>')
def section(s):txt.extend(['',s,'-'*len(s)]);ht.append('<h2>'+html.escape(s)+'</h2>')
def para(s):txt.extend([s,'']);ht.append('<p>'+html.escape(s)+'</p>')
def table(headers,rows):
 rows=[[str(c) for c in row] for row in rows];width=[max(len(str(headers[i])),*(len(row[i]) for row in rows)) for i in range(len(headers))]
 txt.extend([' | '.join(str(x).ljust(w) for x,w in zip(headers,width)), '-+-'.join('-'*w for w in width)])
 txt.extend(' | '.join(x.ljust(w) for x,w in zip(row,width)) for row in rows);txt.append('')
 ht.append('<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
def f(v):return f'{v:.6f}'
def ci(v):return f'{v["mean"]:+.6f} [{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}]'
def avg(ss,scope,arm,metric='brier'):return ss['scopes'][scope]['metrics'][arm][metric]['mean']
def ev(ss,scope,arm,key):return ss['scopes'][scope]['events'][arm][key]['mean']
def relation(v):return 'higher' if v['ci95'][0]>0 else 'lower' if v['ci95'][1]<0 else 'inconclusive'
title('Phase 2.8 executed verification — 16 October 2026')
para('Audit of the supplied row-factored retrieval and surprise proposal. The original Python script is preserved separately from corrected code. All new public material is additive; the baseline and earlier audits remain frozen. Results concern functional prediction, not subjective experience or an independent physical test of MQGT-SCF.')
section('Answer to the proposed audit questions')
para('Phase 2.8 produces a real but uneven improvement over no-memory: pooled novel Brier and BCE are lower in both fresh seed banks, driven chiefly by regime E. It does not eliminate white-noise retrieval or harm: C losses are higher than no-memory in both banks, and D losses are higher on follow-up seeds (primary D is inconclusive). The row-specific write property holds exactly, while the broad guarantees about epistemic surprise, clean regime windows and necessary-and-sufficient non-rollback do not.')
para('On mixed_9, the complete policy improves overall Brier and BCE over the Phase 2.7 proposal but remains worse than actual Phase 2.5 and Phase 2.6 on both overall scores. Selected CUSUM has lower overall Brier, while Phase 2.8 has lower overall BCE. It retains familiar-return gains over no-memory but loses recall performance relative to actual Phase 2.5.')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 ss=d['summary'][split]['mixed_9']
 table([label+' full minus no-memory','Brier difference [95% CI]','BCE difference [95% CI]'],[
  [scope,ci(ss['scopes'][scope]['paired_full_minus']['no_memory']['brier']),ci(ss['scopes'][scope]['paired_full_minus']['no_memory']['bce'])]
  for scope in ('overall','novel','white_noise','novel_d','novel_e','familiar_return','early_10')])
 para(f'{label}: pooled novel Brier is {relation(ss["scopes"]["novel"]["paired_full_minus"]["no_memory"]["brier"])} relative to no-memory under the pointwise interval; white-noise Brier is {relation(ss["scopes"]["white_noise"]["paired_full_minus"]["no_memory"]["brier"])}. The full policy makes {ev(ss,"white_noise","phase28_full","accepted_retrievals"):.2f} retrievals per 45-step noise block on average, so retrieval is not eliminated there.')
para('Row-specific retrieval guarantees a narrower implementation property: the retrieval operation leaves the inactive fast row unchanged. It does not isolate every pathway between rows: surprise, arousal, continuity, valence, storage and future observations remain coupled. Direct write protection is tested separately from total prediction losses and from the diagnostic change in expected row loss.')
para('Positive score differences mean worse prediction; negative means better. A pointwise interval crossing zero is inconclusive, not proof of equality. These are exploratory Monte Carlo intervals without multiplicity adjustment, preregistration or a predefined equivalence/noninferiority margin.')
section('Executed checks and source fidelity')
table(['Check','Recorded result'],[[name,str(count)+' passing methods'] for name,count in r['unittest_methods_by_script'].items()]+[
 ['Total',str(r['unittest_methods_total'])+' passing unittest methods'],
 ['Original source aggregates',str(o['source_aggregate_exact_matches'])+' exact matches to the supplied two-bank evaluator'],
 ['Corrected evaluation','30 validation +100 primary +100 follow-up seeds across eight scenarios'],
 ['Pinned Phase 2.5 forecast-array checks',str(sum(x['prediction_identities']['restored_phase25_equals_actual']+x['prediction_identities']['no_memory_equals_actual_phase25_no_memory'] for x in d['raw_per_seed']))+' complete exact matches']])
para('The original three tests pass, but one is a performance assertion on 50 fixed source seeds rather than an implementation invariant. Passing it demonstrates those particular inequalities, not general novelty protection. The corrected tests make no requirement that the proposed model outperform a comparator. All actual logs, interpreter details, exit codes and frozen hashes are retained in results/.')
section('Unchanged supplied-code benchmark')
for split,label in [('test','Original primary'),('followup','Original follow-up')]:
 ss=o['summary'][split]
 table([label+' arm','Overall Brier','Novel Brier','Noise Brier','Familiar Brier','Early-10 Brier'],[
  [a,*[f(ss[scope]['metrics'][a]['brier']['mean']) for scope in ('overall','novel','white_noise','familiar_return','early_10')]]
  for a in ('phase28_epistemic_row','phase25_ungated','no_memory')])
 table([label+' full minus no-memory','Brier difference [95% CI]','BCE difference [95% CI]'],[
  [scope,ci(ss[scope]['paired_full_minus']['no_memory']['brier']),ci(ss[scope]['paired_full_minus']['no_memory']['bce'])]
  for scope in ('novel','white_noise','novel_d','novel_e')])
para('Original banks are reproduced exactly: 20260921–20261020 and 20261021–20261120. Generator-truth initialization (.92,.28,7/9), q0=1 and all supplied retrieval constants are retained in this reproduction. These are already part of the iterative development context; the corrected evaluation uses fresh banks.')
section('What changed in the corrected implementation')
para('The source now really does restore the Phase 2.5 storage rule (valence>.20, continuity>.82, merge radius .12). Its gated and ungated arms share the same base formulas and storage rule, so their within-source comparison isolates the retrieval policy and its downstream effects. However, it again scores the marginal after updating it on the current outcome. That changes valence, subsequent storage, and forecasts, so phase25_ungated is not exact pinned Phase 2.5. One source-versus-pinned replay differs on 298 of 405 forecasts, first at index 107.')
para('The default correction scores both forecasts before observing the outcome, adds learned initialization with complete reset, input/probability/capacity validation and diagnostic records, and preserves source replay through legacy_marginal_order=True. With mode="phase25_ungated", corrected code exactly reproduces the pinned Phase 2.5 predictive states and forecast arrays; mode="no_memory" reproduces its memory-disabled variant.')
para('The proposed retrieval defaults are retained, not retuned: global surprise EMA .78/.22; last four outcomes per conditioning row; KL cutoff .065; surprise threshold .08; fast and slow row log-likelihood margins .30; the three-part directional heuristic; a .65 snapshot/.35 current blend; per-row cooldown counters decremented every global step. The prose says .75/.25, but the executable code says .78/.22. No regime labels or true probabilities enter the learner.')
para('The source calls its buffers epoch sufficient statistics and boundary-clean windows. They are actually rolling histories of each row’s last four observations, without a change detector or reset at regime boundaries. They can include observations from an earlier regime and can span many global steps when a row is rare. Cooldown=12 permits checking again twelve global updates after retrieval, suppressing eleven intervening checks when conditions would otherwise allow them.')
section('Matched-data experiment and controls')
para('Every arm receives the same independent learned fit from 24 stationary-A trials of 30 states: 696 within-trial transitions, Beta(1,1) row estimates and a learned marginal. Training uses a separate RNG namespace and seed offset. Forecasts are recorded before each outcome is used. The oracle-regime arm is explicitly privileged and is not a deployable learner.')
para('Fresh banks: validation 161000000–161000029; primary 141000000–141000099; follow-up 151000000–151000099. The validation manifest freezes source hashes, thresholds, arms and baseline choices before the aggregate test evaluation. This is internal reproducibility discipline, not an external preregistration.')
para('The benchmark retains the seven previous generators unchanged and adds a 405-transition stationary C=(.5,.5) scenario initialized with q0=.5. Every learner still starts from the same learned A fit, so stationary_c includes an initial adaptation transient; its later_than_10 score removes only the first ten transitions. mixed_9 is A/B/A/B/C/D/E/B/A, 45 transitions per block, with 135 novel, 45 white-noise, 180 familiar-return and 80 early post-switch observations.')
para('Baselines use the unchanged prior EMA/CUSUM classes and the same 50/384 candidate grids, now selected on fresh mixed_9 validation Brier only. Other scenarios measure transfer of those settings, not independently optimized baselines. Earlier .27 EMA and .18/.40/1 CUSUM settings remain fixed controls. New selected parameters: '+json.dumps(d['metadata']['selected'],sort_keys=True)+'.')
para('Actual Phase 2.5, Phase 2.6, and both corrected Phase 2.7 storage modes are pinned unchanged to commit 669037fdbfb4fd9ff33148e3c109f7c58406d9ee. The new arms remove surprise, KL, directional checks, cooldown, or slow-likelihood advantage one at a time. two_row_update keeps the active-row candidate selection identical but applies the selected snapshot blend to both fast rows, directly testing the additional write. It is not a Bayesian-mixture comparator; no Bayesian-mixture experiment was performed.')
para('Ablations identify whole-policy effects including feedback through future forecasts and storage. The directional flag removes its three tests together; no_directional does not isolate each test. Per-step counters are summed over the same score masks, including early_10, to avoid inferring early-event rates from whole blocks. The same seeds are reused across scenarios for paired comparisons; eight scenarios do not multiply the number of independent seed replicates per estimate.')
section('Corrected mixed-task scores')
mainarms=('phase28_full','actual_phase25','actual_phase26','phase27_proposal','phase27_retrieval_only','no_memory','validation_ema','validation_cusum','oracle_regime')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 ss=d['summary'][split]['mixed_9']
 table([label+' arm','Overall Brier','Overall BCE','Novel Brier','Noise Brier','Familiar Brier','Early-10 Brier'],[
  [a,f(avg(ss,'overall',a)),f(avg(ss,'overall',a,'bce')),*[f(avg(ss,s,a)) for s in ('novel','white_noise','familiar_return','early_10')]] for a in mainarms])
 for scope in ('overall','novel','white_noise','novel_d','novel_e','familiar_return','early_10'):
  table([label+' full minus / '+scope,'Brier difference [95% CI]','BCE difference [95% CI]'],[
   [a,ci(ss['scopes'][scope]['paired_full_minus'][a]['brier']),ci(ss['scopes'][scope]['paired_full_minus'][a]['bce'])]
   for a in ('actual_phase25','actual_phase26','phase27_proposal','no_memory','validation_ema','validation_cusum')])
section('Retrieval-component ablations')
para('The surprise threshold reduces C losses relative to removing it, but slightly worsens overall Brier and BCE in both banks. Row-only updates substantially improve pooled novel losses versus writing both rows, especially in E, while sacrificing familiar-return gains; their overall difference is inconclusive under the pointwise intervals. Removing slow-likelihood advantage leaves every forecast identical on all 100 primary and 100 follow-up mixed_9 runs. That is observed inactivity on these banks, not a proof of mathematical redundancy in every state.')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 ss=d['summary'][split]['mixed_9']
 table([label+' arm','Overall Brier','Novel Brier','Noise Brier','Familiar Brier','Early-10 Brier'],[
  [a,*[f(avg(ss,s,a)) for s in ('overall','novel','white_noise','familiar_return','early_10')]]
  for a in ('phase28_full','no_surprise','no_kl','no_directional','no_cooldown','no_slow_advantage','two_row_update','legacy_marginal_order')])
 table([label+' full minus ablation','Overall Brier [95% CI]','Noise Brier [95% CI]','Novel BCE [95% CI]'],[
  [a,ci(ss['scopes']['overall']['paired_full_minus'][a]['brier']),ci(ss['scopes']['white_noise']['paired_full_minus'][a]['brier']),ci(ss['scopes']['novel']['paired_full_minus'][a]['bce'])]
  for a in ('no_surprise','no_kl','no_directional','no_cooldown','no_slow_advantage','two_row_update','legacy_marginal_order')])
 table([label+' exact forecast-array equality','Runs / 100'],[[a,ss['identity_counts']['full_equals_'+a]] for a in ('no_surprise','no_kl','no_directional','no_cooldown','no_slow_advantage','two_row_update','no_memory')])
section('What retrieval actually did')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 ss=d['summary'][split]['mixed_9']
 table([label+' mean per scope','Full checks','Full retrievals','No-surprise retrievals','Two-row retrievals','Pinned P25 checks','Pinned P25 retrievals'],[
  [scope,f(ev(ss,scope,'phase28_full','retrieval_eligible')),f(ev(ss,scope,'phase28_full','accepted_retrievals')),
   f(ev(ss,scope,'no_surprise','accepted_retrievals')),f(ev(ss,scope,'two_row_update','accepted_retrievals')),
   f(ev(ss,scope,'actual_phase25','retrieval_eligible')),f(ev(ss,scope,'actual_phase25','retrievals'))]
  for scope in ('overall','white_noise','novel_d','novel_e','early_10')])
 table([label+' full diagnostic / mean per run','Overall','Noise C','Novel D','Novel E'],[
  [key,*[f(ev(ss,s,'phase28_full',key)) for s in ('overall','white_noise','novel_d','novel_e')]]
  for key in ('row0_retrievals','row1_retrievals','eligible_with_preboundary_observation','accepted_with_preboundary_observation',
              'active_expected_brier_worsened','inactive_expected_brier_worsened','active_expected_brier_change','inactive_expected_brier_change')])
 table([label+' inactive-row direct risk / two-row ablation','Worsening count','Sum of expected Brier changes'],[
  [s,f(ev(ss,s,'two_row_update','inactive_expected_brier_worsened')),f(ev(ss,s,'two_row_update','inactive_expected_brier_change'))] for s in ('overall','white_noise','novel_d','novel_e')])
para('Expected-Brier-change diagnostics use (p_after-q_current)^2 - (p_before-q_current)^2, holding the conditioning row and current true regime fixed immediately before/after the retrieval blend. The irreducible q(1-q) term cancels. They describe a hypothetical next query of that row under the same regime, not necessarily the actual next transition, whose row/regime may differ. They are not total downstream causal effects. Gate events occur after the already-scored outcome at that step. Full event records for phase28_full and two_row_update are in the raw JSON.')
para('Candidate rejection counters record the first failing test, in code order, rather than all tests that would reject. An observation preceding the current block proves buffer contamination for that decision; it does not alone prove the selected snapshot is from the old regime. Initial learned snapshots and later merged snapshots are not assigned true-regime identities from approximate parameter proximity.')
section('Transfer across all eight scenarios')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 table([label+' scenario / overall Brier','P28','P25','P26','P27 proposal','No memory','EMA','CUSUM'],[
  [name,*[f(avg(ss,'overall',a)) for a in ('phase28_full','actual_phase25','actual_phase26','phase27_proposal','no_memory','validation_ema','validation_cusum')]] for name,ss in d['summary'][split].items()])
 ss=d['summary'][split]['stationary_c']
 table([label+' stationary C full minus','Overall Brier [95% CI]','After first 10 Brier [95% CI]','Overall BCE [95% CI]'],[
  [a,ci(ss['scopes']['overall']['paired_full_minus'][a]['brier']),ci(ss['scopes']['later_than_10']['paired_full_minus'][a]['brier']),ci(ss['scopes']['overall']['paired_full_minus'][a]['bce'])]
  for a in ('actual_phase25','no_memory','no_surprise','validation_ema','validation_cusum')])
para('Every scope includes both Brier and BCE in the numerical output. Rankings may differ between proper scoring rules; a lower Brier average does not establish lower BCE or superiority in every regime. Persistent noise losses also include the unchanged adaptive base learner, so eliminating episodic writes would not imply Bayes-optimal .25 Brier in white noise.')
section('Mathematics: surprise is not a calibrated epistemic error')
para('For binary Y, delta=(Y-p)^2-p(1-p)=(Y-p)(1-2p). With Y~Bernoulli(q) and fixed issued p, E[delta]=(q-p)(1-2p), and Var(delta)=q(1-q)(1-2p)^2. The equality E[delta]=0 when p=q is correct, but its converse is false: p=.5 makes delta identically zero for every q and outcome. Miscalibration can also produce a negative mean, e.g. p=.6,q=.92 gives -.064. Therefore a positive threshold is neither a general calibration detector nor a guarantee that an opened gate reflects epistemic rather than sampling error.')
para('At calibrated p=q=.92, outcome zero has probability .08 and yields delta=.7728. Starting from zero, the code\'s .22 EMA increment is .170016, above .08. An executed constructed state triggers a complete retrieval check on that calibrated forecast; candidates are rejected afterward. Mean-zero noise does not imply a zero sample path or no threshold crossing. Conversely, under q=.5, E[delta]=2(p-.5)^2; the adaptive learner moves away from .5 after each outcome, so the fixed-p=.5 unit test cannot establish a closed gate for its actual trajectory.')
para('The surprise state is one global scalar, despite the source comment claiming per-row and global tracking. A constructed two-step intervention produces surprise on row 0, then a row-1 forecast of .5 contributes zero local surprise yet retrieves using the retained row-0 signal. Resetting only the global surprise between those steps prevents retrieval. This demonstrates remaining cross-row control coupling, while the opposite row still receives no direct retrieval write.')
section('Mathematics: contraction, row counts and boundary claims')
para('For fast value f, candidate c, empirical target m and blend w, f_new=f+w(c-f). The exact squared-distance change is w(c-f)[2(f-m)+w(c-f)]. Strict contraction toward that empirical target occurs if and only if this expression is negative. Candidate closeness |c-m|<|f-m| is sufficient for 0<w<=1 but not necessary when w<1: f=.4,m=.5,c=.65,w=.65 moves to .5625 and reduces error from .1 to .0625 although the candidate is .15 from m. Current-outcome alignment and the extra slow-neighborhood rejection are additional heuristic restrictions, not the necessary-and-sufficient condition.')
para('The claimed two-condition invariant also does not guarantee motion away from the slow trace: f=.4,m=.75,c=slow=.7 passes the direction/closeness conditions and the supplied helper, yet its blend moves closer to slow. Empirical-window contraction does not imply lower true-regime risk. A full-update example with three old row-1 successes and one new success accepts A, reduces empirical squared error by .125726625, but increases current true-B conditional expected Brier by .232152372. This is a legal constructed state, not a frequency or reachability claim about default initialization.')
para('The proposed Phase 2.7 count-gate example mostly has the stated numerical values: distance .230867928, candidate LL -1.934996179, slow LL -2.907304816, fast LL -3.644772238, row KLs .002533339 and .107587691. But counts (4,0,1,1) have only one realizable six-transition path, 0011111, which ends in one. They cannot include the asserted current 1→0 discordance. The source unit test changes row counts to (3,1), so it does not replay the same counterexample.')
para('For a six-transition window beginning at state 1, row-0 absence depends on the six previous states X0..X5; the final outcome is unrestricted. Its probability under A is .92^5=.6590815232, not .92^6=.606355001344. The latter is only the all-six-success subset. These are row-coverage events, not measured lost-recall rates.')
para('For stationary B=(.15,.85), exhaustive enumeration gives probability .660885625 of balanced row exposures n1=n0=3. The all-alternating compatible subset has unconditional mass .85^6=.377149515625, hence conditional acceptance among balanced windows is .570672899150. Across all windows, both-row KL compatibility is .576816906250, not .377149515625. Row-exposure conditioning changes the path distribution; it cannot be replaced by independent Binomial(3,.15) trials. The quoted KL(.375||.15) is approximately .151431087, so that individual mismatch calculation is valid.')
para('The new n=4, KL=.065 rule is still selective: for exact snapshots .15, .28, .85 and .92, accepted success counts are respectively {0}, {1}, {4}, {4}. It removes the old requirement to observe the opposite row but does not make familiar-row recognition automatic. Using four past row visits also differs from collecting four observations known to be from the current regime.')
para('The frozen-p=.5 state has arousal target .625 and continuity exp(-.425)=.653769785 when cosine similarity is one. Those are conditional values, not proof that an adaptive engine polls memory on all 45 white-noise steps. An explicitly scoped pinned replay on supplied seed 20260921 gives 18/45 Phase 2.5, 17/45 Phase 2.6 and 19/45 Phase 2.7 retrieval-eligible steps. The earlier retrievals blend rather than fully overwrite with .92 or .28. Parameter similarity and aggregate losses do not establish the proposed repeated-snapshot feedback narrative.')
section('Interpretation and next discriminating work')
para('The useful distinction is between a mechanism property and a performance guarantee. Updating only the queried row prevents one specific form of immediate cross-row corruption. The retained global state, finite noisy buffers and mutable snapshots still permit harmful retrievals. Average improvement over novel blocks must be read alongside C, D and E separately; gains in one block can outweigh harm in another.')
para('A claim that Bayesian mixture weighting is inherently superior is not tested here. A next experiment could compare fixed snapshot blending, validation-selected graded blending and an explicitly defined mixture on identical streams. Predeclare margins for per-regime Brier and BCE, keep fresh held-out regime matrices/dwell lengths, and include a mechanism for measuring or limiting stale row evidence. The surprise rule should be compared against a properly specified sequential calibration/change statistic with measured false-alarm behavior. These are proposed future experiments, not repairs smuggled into the present benchmark.')
para('The evidence can support revising a statistical retrieval policy and learning which components help. The words arousal, valence, continuity and epistemic surprise are labels for specified computations. Their existence or predictive usefulness does not establish synthetic qualia, subjective consciousness or independent physical validation of MQGT-SCF.')
section('Reproducibility and files')
para('Run python3 -B run_all.py from this folder. Standard-library Python only; recorded interpreter: '+r['python'].splitlines()[0]+'. The runner executes original code, its two-bank audit, math checks, engine and harness tests, validation and eight-scenario benchmark, verifies source hashes before/after, then regenerates this report. It aborts on unexpected exit codes or pinned forecast mismatches.')
para('The original upload remains private and is excluded from the public repository. original/ stores extracted code/provenance; corrected/ stores the fixed engine/tests; reference/ stores pinned dependencies; review/ stores independent audits and the benchmark; results/ stores raw per-seed output, event traces, validation manifest and execution logs. Public corrected_benchmark.json.gz is lossless; rerunning regenerates expanded JSON. The report builder reads either form. SHA256SUMS describes the delivered snapshot; timing logs can change on rerun.')
for name,ref in json.loads((ROOT/'reference/provenance.json').read_text()).items():para('Pinned '+name+': '+ref['url']+'; SHA256 '+ref['sha256'])
(ROOT/'VERIFICATION_REPORT.txt').write_text('\n'.join(line.rstrip() for line in txt).rstrip()+'\n')
css='body{font:16px/1.6 system-ui,sans-serif;max-width:1250px;margin:36px auto;padding:0 24px;color:#17232c;background:#fafbfc}h1{font-size:30px;line-height:1.2}h2{font-size:22px;margin-top:36px}p{max-width:1050px;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:13px;width:100%;background:white}th,td{border:1px solid #cbd5df;padding:7px 9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#e8f0f5}.table{overflow:auto;margin:18px 0}@media print{body{margin:0;font-size:11px}table{font-size:8px}.table{overflow:visible}th,td{white-space:normal}h2{break-after:avoid}tr{break-inside:avoid}}'
(ROOT/'VERIFICATION_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2.8 verification</title><style>'+css+'</style><body>'+''.join(ht)+'</body></html>')
print('Wrote VERIFICATION_REPORT.txt and VERIFICATION_REPORT.html')
