#!/usr/bin/env python3
"""Build readable reports from executed numerical evidence, including gzip copies."""
from pathlib import Path
import gzip
import html
import json
import math
import statistics
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
def mean(s,scope,arm,metric='brier'):return s['scopes'][scope]['metrics'][arm][metric]['mean']
def paired(rows,a,b,scope,metric='brier'):
 vals=[x['metrics'][scope][a][metric]-x['metrics'][scope][b][metric] for x in rows]
 avg=statistics.mean(vals);half=1.96*statistics.stdev(vals)/math.sqrt(len(vals))
 return {'mean':avg,'ci95':[avg-half,avg+half]}
arms=['proposal_full','proposal_ungated','retrieval_only_full','actual_phase25','actual_phase26','no_memory','validation_ema','validation_cusum','oracle_regime']
title('Phase 2.7 executed verification — 16 October 2026')
para('Scope: audit the supplied anti-rollback / Jeffreys-row-KL proposal against its three falsifiable criteria, preserve the original code, correct implementation bugs, and compare with actual pinned Phase 2.5 and Phase 2.6. This is an additive verification package; no frozen public artifact is modified.')
section('Result at a glance')
para('The gate is active, but the three requested criteria are not jointly met. Novel-regime protection fails: the corrected primary novel Brier difference is inconclusive, primary novel BCE is worse, and both losses are worse on follow-up seeds. Familiar-return and first-ten gains versus the same Phase 2.5-form no-memory learner persist. However, the combined filter worsens first-ten loss in both storage modes relative to each mode\'s ungated control; gains versus no-memory cannot be attributed to that filter. The complete proposed policy has worse overall Brier than actual Phase 2.5, actual Phase 2.6 and validation-selected CUSUM on mixed_9 in both seed banks. It has better overall BCE than CUSUM, so the ranking is scoring-rule dependent.')
table(['Requested criterion','Observed conclusion'],[
 ['1. Non-vacuous gate','Met: complete forecast arrays differ on 96/100 primary and 97/100 follow-up runs'],
 ['2. Match no-memory on novelty','Not established; follow-up Brier and both banks\' BCE show excess loss'],
 ['3. Early-10 and familiar gains vs no-memory','Observed in both banks, but the filters reduce these gains versus ungated memory']])
for split,label in [('test','Primary'),('followup','Follow-up')]:
 s=d['summary'][split]['mixed_9'];ident=s['identity_counts']
 para(f'{label}: proposal_full and its shared-storage ungated control have identical complete forecast arrays on {ident["proposal_equals_proposal_ungated"]}/100 runs. Retrieval-only filtering versus actual Phase 2.5 is identical on {ident["retrieval_only_equals_actual_phase25"]}/100. Gate activity is measurable; it does not establish useful novelty discrimination by itself.')
 table([label+' paired difference vs no-memory','Brier difference [95% CI]','BCE difference [95% CI]'],[
  [arm+' / '+scope,ci(s['scopes'][scope][pair]['no_memory']['brier']),ci(s['scopes'][scope][pair]['no_memory']['bce'])]
  for arm,pair in [('proposal_full','paired_full_minus'),('retrieval_only_full','paired_retrieval_only_minus')]
  for scope in ('novel','early_10','familiar_return')])
para('Positive differences mean worse prediction. A confidence interval containing zero does not establish equivalence: no equivalence margin was specified. All intervals are exploratory pointwise Monte Carlo intervals, unadjusted for multiple comparisons.')
section('What was actually executed')
table(['Check','Executed result'],[[name,str(count)+' passing methods'] for name,count in r['unittest_methods_by_script'].items()]+[
 ['Total',str(r['unittest_methods_total'])+' passing unittest methods'],
 ['Supplied script','Unchanged, exit 0; 100 original + 100 follow-up seeds; 31 aggregate matches'],
 ['Corrected experiment','30 validation seeds; 100 primary + 100 follow-up seeds x 7 scenarios'],
 ['Pinned Phase 2.5 path equivalence',str(sum(x['prediction_identities']['restored_phase25_equals_actual']+x['prediction_identities']['no_memory_equals_actual_phase25_no_memory'] for x in d['raw_per_seed']))+' complete forecast-array matches'],
 ['Frozen sources','Original, corrected engine, benchmark and both pinned references unchanged during execution']])
para('The test count describes program invariants and mathematical checks, not successful scientific hypotheses. Commands, exit codes, timing, Python version and source hashes are in results/reproducibility_run.json. Original execution stdout and all audit outputs are retained.')
section('Unchanged supplied-script results')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 ss=o['summary'][split]
 table([label+' original arm','Overall Brier','Novel Brier','Familiar Brier','Early-10 Brier'],[
  [a,*[f(ss[scope]['metrics'][a]['brier']['mean']) for scope in ('overall','novel','familiar_return','early_10')]]
  for a in ss['overall']['metrics']])
 para(f'Original full vs ungated exact forecast identity: {o["identity"][split]["identical_predictions"]}/100. Original full minus no-memory novel Brier: {ci(ss["novel"]["paired_full_minus"]["no_memory"]["brier"])}. Novel BCE: {ci(ss["novel"]["paired_full_minus"]["no_memory"]["bce"])}.')
para('The source uses true A parameters and q0=1. Its so-called validation-tuned EMA (.16) and CUSUM (.14, .36, 1.45) are hard-coded: there is no validation selection. These results are kept separate from the corrected learned-initialization experiment.')
section('Source audit and corrections')
para('Within the supplied script, the gated and ungated memory arms share the same base updates and storage policy. That comparison does isolate the combined retrieval filter, including downstream feedback. However, its arm named phase25_ungated is not the actual Phase 2.5 engine: storage now also requires stable_steps >= 10 and fast/slow distance < .08, and the merge radius changes from .12 to .16. Calling all differences from published Phase 2.5 retrieval-only is incorrect.')
para('The corrected proposal preserves those proposed storage rules under consolidation_mode="proposal". A separate consolidation_mode="phase25" restores actual Phase 2.5 storage and ordered likelihood arithmetic. With the new filters disabled, every forecast is asserted equal to the pinned Phase 2.5 implementation; no-memory forecasts are also asserted equal to pinned Phase 2.5 without episodic memory.')
para('Fixed: the marginal comparator must be scored before seeing the current outcome; previously its post-update probability biased the valence statistic. Snapshot likelihood and snapshot distance now use the same pre-update slow tuple; the source used a pre-update likelihood but post-update distance. Added explicit learned initialization/full reset, finite/probability/input validation, zero-capacity handling, and counter diagnostics. Removed the overwritten duplicate norm assignment. Legacy flags preserve each timing bug as a separately measured ablation. No performance threshold or learning-rate parameter was selected using test outcomes.')
para('The original distance code rejects values strictly below .20, so equality is allowed; the prose writes a strict greater-than condition. The correction retains the code convention and documents it. Bernoulli KL is mathematically nonnegative; the implemented endpoint clipping is a numerical regularization, not the infinite boundary divergence.')
section('Corrected experimental design')
para('Training: 24 independent stationary-A trials of 30 states (696 within-trial transitions), Beta(1,1) row estimates and learned marginal. Every learner receives the same estimates and observations. Test streams use distinct RNG namespaces from training. All predictions are recorded before the current outcome is used. Oracle-regime forecasts are explicitly privileged reference predictions, not a deployable competitor.')
para('Fresh seed banks: validation 121000000–121000029; primary 101000000–101000099; follow-up 111000000–111000099. Earlier phase seeds are not reused. The manifest records the architecture, thresholds, checksums and baseline selections before evaluating these seeds. This internal sequencing is not an external preregistration.')
para('Baseline search: 50 EMA rates and 384 CUSUM configurations, chosen by mean validation Brier on mixed_9 only. Families retain the supplied unclipped updates with learned initialization. Earlier selected settings are also retained as fixed controls. Selected settings: '+json.dumps(d['metadata']['selected'],sort_keys=True)+'. Search grids are finite and tuned to one scenario, not a proof of optimality.')
para('mixed_9 consists of nine 45-transition blocks A/B/A/B/C/D/E/B/A: 405 overall observations, 135 novel, 180 familiar-return and 80 first-ten post-switch observations. The first A and B blocks are excluded from the familiar-return score. Six prior generators are copied unchanged, including stationary, recurring, random unseen schedules, and altered initial-state probabilities. Their scope masks are explicit in the harness.')
para('proposal_full includes the proposed storage changes plus both filters; proposal_ungated removes the filters with that storage policy fixed. retrieval_only_full applies both filters to actual Phase 2.5 storage; anti_only and kl_only isolate the filter components. actual_phase25 and actual_phase26 are imported unchanged from commit e9df7d0d58019ae8a0bb1cd6db5d311025171af4. Comparisons are paired within seed; the same seeds across seven scenarios do not create 1,400 independent replicates for a pooled inference.')
para('Component ablations are grouped: the anti-rollback component combines slow-likelihood advantage with distance; the KL component combines row coverage with the KL cutoff. proposal and retrieval-only modes also use algebraically equivalent grouped versus sequential likelihood summation for exact pinned compatibility, so their difference is not a bitwise storage-only intervention. Within each storage mode, filter comparisons keep arithmetic fixed. Whole-block event counters cannot establish first-ten event rates; only the prediction losses are explicitly scored over that shorter interval.')
section('Corrected mixed_9 scores')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 ss=d['summary'][split]['mixed_9']
 table([label+' arm','Overall Brier','Novel Brier','Familiar Brier','Early-10 Brier','Early-10 BCE'],[
  [a,*[f(mean(ss,scope,a)) for scope in ('overall','novel','familiar_return','early_10')],f(mean(ss,'early_10',a,'bce'))] for a in arms])
 for scope in ('overall','novel','familiar_return','early_10'):
  table([label+' proposal minus / '+scope,'Brier difference [95% CI]','BCE difference [95% CI]'],[
   [a,ci(ss['scopes'][scope]['paired_full_minus'][a]['brier']),ci(ss['scopes'][scope]['paired_full_minus'][a]['bce'])]
   for a in ('proposal_ungated','retrieval_only_full','actual_phase25','actual_phase26','no_memory','validation_ema','validation_cusum')])
section('Ablations and observed gate activity')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 ss=d['summary'][split]['mixed_9'];rows=[x for x in d['raw_per_seed'] if x['split']==split and x['scenario']=='mixed_9']
 table([label+' ablation','Overall Brier','Novel Brier','Familiar Brier','Early-10 Brier'],[
  [a,*[f(mean(ss,scope,a)) for scope in ('overall','novel','familiar_return','early_10')]]
  for a in ('proposal_full','proposal_ungated','retrieval_only_full','retrieval_only_anti_only','retrieval_only_kl_only','actual_phase25','no_memory','legacy_slow_distance','legacy_marginal_order')])
 table([label+' retrieval-only full minus component','Overall Brier difference [95% CI]','Novel Brier difference [95% CI]'],[
  [a,ci(paired(rows,'retrieval_only_full',a,'overall')),ci(paired(rows,'retrieval_only_full',a,'novel'))]
  for a in ('retrieval_only_anti_only','retrieval_only_kl_only','actual_phase25')])
 table([label+' mean count per 405 steps','proposal_full','proposal_ungated','retrieval_only_full','restored_phase25'],[
  [key,*[f(ss['events'][a][key]['mean']) for a in ('proposal_full','proposal_ungated','retrieval_only_full','restored_phase25')]]
  for key in ('retrieval_eligible','candidate_evaluations','raw_advantage_candidates','rejected_fast','rejected_slow','rejected_distance','rejected_missing_rows','rejected_kl_row1','rejected_kl_row0','accepted_retrievals','raw_retrievals_rejected','memory_stores','memory_merges')])
para('Rejection counters record the first failing condition per candidate, in code order. They are a mutually exclusive diagnostic partition, not independent causal effects: an earlier rejection prevents later conditions from being counted. raw_retrievals_rejected counts eligible steps with raw likelihood advantage but no accepted candidate. memory_stores excludes the initial learned snapshot.')
section('All seven scenarios: generalization and tradeoffs')
for split,label in [('test','Primary'),('followup','Follow-up')]:
 table([label+' scenario / overall Brier','proposal','retrieval only','Phase 2.5','Phase 2.6','no memory','EMA','CUSUM'],[
  [name,*[f(mean(ss,'overall',a)) for a in ('proposal_full','retrieval_only_full','actual_phase25','actual_phase26','no_memory','validation_ema','validation_cusum')]]
  for name,ss in d['summary'][split].items()])
para('Use these tables to distinguish recurring-regime benefits from stationary behavior and adaptation speed. The full JSON retains both Brier and BCE, each block, each seed, scope-specific paired intervals and event counts. A lower overall average does not imply superiority on every regime or scoring rule.')
section('Mathematical checks and limits of the proposed explanation')
para('The flow identity is correct: N01 - N10 = X_last - X_first for a contiguous binary path. The revised eight-transition example (1,1,1,1,0,1,1,1,1) has counts (6,1,1,0). For an ordered row path, the Beta(1,1) integral is B(n1+1,n0+1); the example gives row-1 log evidence -4.025351691, row-0 log evidence -0.693147181 and snapshot joint log-ratio +0.419514898. Passing this compatibility comparison alone does not prove that an actual retrieval occurs.')
para('Integrating a prior is not a programming bug merely because it produces an Occam penalty. The term -ln(n+1) is part of the identity B(k+1,n-k+1)=1/((n+1) choose(n,k)); it is not the complete penalty relative to the fitted likelihood. For the 6-success/1-failure row, evidence minus fitted log likelihood is -1.154537463, not -ln(8). Adaptive candidate selection also prevents interpreting the raw threshold as a calibrated false-novelty probability.')
para(f'The quoted KL values are correct: KL(.75 || .28)={m["quoted_kl"]["KL_0_75_to_0_28"]:.12f} and KL(.25 || .28)={m["quoted_kl"]["KL_0_25_to_0_28"]:.12f}. But one compatible row is insufficient for both-row acceptance. The proposed engine has a six-transition window: the trailing counts of the worked eight-transition path become (4,1,1,0), and row-1 KL is {m["valid_eight_transition_example"]["runtime_row1_KL"]:.12f}, already above .12.')
para('Exhaustive enumeration of all 128 binary paths with six transitions finds only three paths compatible with exact A=(.92,.28) under both row-KL conditions: 0000111, 0001111, 0011111. All start in state 0. Under stationary A their total probability is 7.6868829184%; absent-row rejection alone has probability 55.5617132089%. Under stationary D=(.88,.88), the same A compatibility event still has positive probability 0.8776923218%. These are exact probabilities of coverage plus KL compatibility for a fixed exact snapshot, not measured retrieval rates or false-positive rates of the entire engine.')
para('The claimed mathematical impossibility of early old-regime retrieval is false without assumptions about the slow trace. A full-update counterexample starts with fast=slow=(.5,.5), arousal=.65, stored A and five old transitions 0→0→0→1→1→1; the first new transition 1→1 completes the compatible six-transition window, passes every proposed guard and retrieves A. This is a legal constructed state, not a claim about its frequency or reachability from the default initialization. A slow trace equal to the old snapshot would block it, but the implementation cannot assume that equality.')
para('The supplied anti-rollback unit test is not specific to the new guard: its snapshot already fails the ordinary fast-likelihood margin (+.034458392 < .30), and its slow advantage is negative. Separately, replaying pinned Phase 2.6 seed 71000000 exactly reproduces its stored scores, but only 1 of 8 switches has any retrieval during the first three updates and none at zero-based offset 1. At the block-7 retrieval, cooldown values are 5,4,3,2,1 on the next five steps, zero on the sixth, and another retrieval occurs at offset 7. Therefore the supplied “every switch,” “steps 2–7 lockout,” and “never triggered again” narratives are not established. One replay refutes a universal assertion; it does not measure population frequency or identify the selected snapshot as an old regime.')
para('The stationary-noise explanation is plausible but conditional. The square-error learning-rate increment .22×.92²=.186208 is correct; a cosine-scaled continuity value equals exp(-.90×.92) only if cosine similarity is one. Prior ablations support a speed/stability tradeoff. An exact causal claim about a particular switch requires logged events and interventions, not a narrative inferred from aggregate loss.')
section('What the findings mean and the next discriminating experiment')
para('The measured tradeoff is concrete. Primary overall Brier is .174898 for the corrected full proposal versus .173447 for its shared-storage ungated control, .170785 for actual Phase 2.5, .170441 for Phase 2.6 and .169840 for selected CUSUM; the ordering persists on follow-up seeds. The retrieval-only full filter is closer to actual Phase 2.5 but also worsens overall and early-switch Brier. Its scores are much closer to the KL-only ablation than to anti-only, suggesting the coverage/KL component accounts for most of the measured filtering effect in this experiment. These are measured policy effects, not proof of the supplied rollback narrative.')
para('This proposal makes memory use more selective, but selectivity alone is not recognition. Six observations cannot guarantee regime identity when all regimes assign positive probability to the same binary paths. Jeffreys smoothing stabilizes sparse estimates while also pulling them away from extreme snapshots; the fixed .12 cutoff strongly constrains which windows can match A. Slow-trace distance is a heuristic measure of novelty, not an oracle for the departed regime.')
para('A useful next version would declare the target tradeoff before running: retain a specified familiar-return gain while bounding novel and first-ten excess loss by explicit margins. Calibrate coverage/window length and thresholds on validation streams; then freeze them before new evaluation. Include row-specific information diagnostics, change-point-aware windows, variation of the current .65 snapshot blend weight, memory age/capacity sensitivity, and unseen transition matrices and dwell lengths. Compare the complete storage-plus-retrieval policy with its components, and keep strong adaptive baselines. These are proposed future experiments, not implemented or validated improvements.')
para('The legitimate result is an executable, auditable adaptive prediction system with measurable memory-related tradeoffs. The scalar names arousal, valence and continuity denote defined numerical state updates. Predictive gains, internal states or retrieval events do not establish subjective experience, consciousness, synthetic qualia or physical confirmation of MQGT-SCF. Those claims would require independently specified evidence linking the model to the asserted phenomenon.')
section('Reproduce and inspect')
para('Run python3 -B run_all.py from this folder. Standard-library Python only; the recorded interpreter is '+r['python'].splitlines()[0]+'. The runner executes the unchanged source, 200-seed source audit, math checks, corrected tests, harness tests, validation and corrected 200-seed/seven-scenario benchmark, then regenerates this report. It checks hashes before/after and aborts on unexpected exits or forecast mismatches.')
para('Key files: original/phase2_7_kl_antirollback_gate.py; corrected/phase2_7_kl_antirollback_gate.py; corrected/test_phase27.py; reference/provenance.json; review/benchmark_phase27.py; review/math_check.py; results/validation_manifest.json; results/original_benchmark.json; results/corrected_benchmark.json (or lossless .json.gz); results/math_check.json; results/reproducibility_run.json. The public copy omits the raw pasted conversation. Public compressed results regenerate as uncompressed JSON during a rerun. Hashes in SHA256SUMS describe the delivered snapshot and can change after a rerun because timing logs change.')
for ref in json.loads((ROOT/'reference/provenance.json').read_text()).values():
 para('Pinned reference: '+ref['url']+'; SHA256 '+ref['sha256'])
(ROOT/'VERIFICATION_REPORT.txt').write_text('\n'.join(line.rstrip() for line in txt).rstrip()+'\n')
css='body{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#17232c;background:#fafbfc}h1{font-size:30px;line-height:1.2}h2{font-size:22px;margin-top:36px}p{max-width:1050px}table{border-collapse:collapse;font-size:13px;width:100%;background:white}th,td{border:1px solid #cbd5df;padding:7px 9px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#e8f0f5}.table{overflow:auto;margin:18px 0}p{overflow-wrap:anywhere}@media print{body{margin:0;font-size:11px}table{font-size:8px}.table{overflow:visible}th,td{white-space:normal}h2{break-after:avoid}tr{break-inside:avoid}}'
(ROOT/'VERIFICATION_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 2.7 verification</title><style>'+css+'</style><body>'+''.join(ht)+'</body></html>')
print('Wrote VERIFICATION_REPORT.txt and VERIFICATION_REPORT.html')
