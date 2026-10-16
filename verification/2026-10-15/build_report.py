"""Render the completed numerical audit as a self-contained HTML and text report."""
from pathlib import Path
import difflib
import hashlib
import html
import json
import re

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def f(x):return f'{x:.6f}'
parts=[];plain=[]
def heading(text,level=2):
    parts.append(f'<h{level}>{html.escape(text)}</h{level}>');plain.extend(['',text,'='*len(text)])
def p(text):
    parts.append('<p>'+html.escape(text)+'</p>');plain.extend([text,''])
def pre(text):
    parts.append('<pre>'+html.escape(text)+'</pre>');plain.extend([text,''])
def link(label,url):
    parts.append(f'<p><a href="{html.escape(url,quote=True)}">{html.escape(label)}</a></p>')
    plain.extend([f'{label}: {url}',''])
def table(headers,rows):
    parts.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    plain.append(' | '.join(headers));plain.extend(' | '.join(map(str,row)) for row in rows);plain.append('')

def main():
    exe=read('results/execution_manifest.json');assert exe['success']
    a=read('results/phase23_original_benchmark.json');b=read('results/phase23_corrected_benchmark.json')
    c=read('results/phase24_original_benchmark.json');d=read('results/phase24_corrected_benchmark.json')
    mathdata=read('results/math_verification.json')
    commit=exe['repository_commit'];source=f'https://github.com/Cbaird26/zoraasi-qualia-research/blob/{commit}/'
    heading('ZoraASI Phase 2.3 / 2.4 — executed verification',1)
    p('15 October 2026 · America/New_York · Local verification of the supplied Gemini text, pinned public source, and separately corrected research scripts.')
    p('Outcome: both uploaded scripts execute and their six supplied tests pass. Phase 2.3 is not an exact reproduction as uploaded: initialization and the derangement differ from the public source. The corrected local version matches the frozen report and source output across 200 seeds. Phase 2.4 demonstrates useful adaptive prediction in the tested switching environments, but the full architecture is not consistently best. No result measures subjective experience or validates MQGT-SCF.')
    heading('What was actually checked')
    original_hash=read('original/extraction_manifest.json')['uploaded_text_sha256']
    table(['Item','Verification'],[
        ['Uploaded text','Two Python fences extracted exactly from the supplied technical response. The raw conversational text is excluded from this public supplement; script hashes are recorded in original/extraction_manifest.json.'],
        ['Attachment SHA-256',original_hash],
        ['Pinned GitHub commit',commit],
        ['Reference integrity','13 Git blob hashes verified; all 7 published SHA-256 checksums pass; snapshot contents unchanged after execution.'],
        ['Runtime',exe['python'].splitlines()[0]+'; '+exe['platform']],
        ['Execution','Every command in results/execution_manifest.json exited 0. Standard library only; no packages installed.'],
        ['Review provenance','Primary Codex execution plus two Codex subagent reviews of mathematics and Phase 2.4. No independent Gemini, Grok, or Claude execution is claimed.']])
    link('Pinned public protocol',source+'PROTOCOL.md');link('Pinned public implementation',source+'zora_lab/connection_study.py')
    rows=[];total=0
    for name,label in [('original_phase23','Uploaded Phase 2.3'),('original_phase24','Uploaded Phase 2.4'),('repository_tests','Public repository'),('corrected_tests','Corrected scripts'),('math_verification_execution','Independent mathematics')]:
        log=(ROOT/'results'/f'{name}.txt').read_text();matches=re.findall(r'Ran (\d+) tests?',log)
        n=int(matches[-1]);total+=n;rows.append([label,n,'PASS'])
    table(['Test suite','Tests','Result'],rows)
    p(f'{total} test cases passed in the packaged end-to-end run. The corrected suite includes source conformance over 200 seeds, causal prefix perturbations, reset/memory isolation, and invalid-input checks. These are software checks; a passing test suite is not independent empirical confirmation of the research claims.')
    heading('Phase 2.3 — exact source comparison')
    table(['Feature','Uploaded implementation','Public source / correction'],[
        ['Training/test size','24 / 12 trials, 30 observations each','Matches: 696 training and 348 test transitions.'],
        ['Parameter fit','Training-only Beta(1,1) smoothing','Matches transition and pooled target-marginal estimators.'],
        ['Trial initialization','random() < 7/9; stationary','getrandbits(1); fair coin. Both distribution and random-number consumption differ.'],
        ['Derangement','Fixed shift of 1','Random(seed+1).randrange(1,12), fixed across time. Shift is 11 for seed 20260921.'],
        ['Four prediction arms','Previous state, marginal, X0, other-trial prior state','All four information conditions correctly implemented; reset is deliberately weak.'],
        ['Same-value fraction','Omitted','Required by protocol; corrected result is 206/348 = 0.591954. Index derangement does not imply unequal binary values.'],
        ['Telemetry B1 checks','Not implemented in uploaded standalone script','All original public telemetry tests run separately and pass; corrected standalone script reproduces the forecasting component.'],
        ['Input validation','Scores silently zip unequal lengths; empty/invalid inputs not consistently rejected','Corrected lengths, binary values, finite probabilities, dimensions, smoothing, and seed-count checks.'],
        ['Reproducibility output','Console metrics only; SD promised but omitted in returned summary','Corrected JSON includes raw data, predictions, parameters, permutation, per-seed scores, SDs and paired intervals.']])
    p('The source protocol’s wording does not fully specify the initialization distribution or cyclic shift. Source inspection resolves both. Gemini’s version remains a valid stationary variant; it must not be labeled an exact reproduction. The corrected script defaults to repository mode and retains the stationary variant as --mode stationary. Neither the public repository nor the extracted original scripts was edited.')
    heading('Phase 2.3 — actual seed 20260921 results')
    table(['Arm','Uploaded Brier','Uploaded BCE','Corrected/public Brier','Corrected/public BCE'],[
        [arm,f(a['single_seed']['metrics'][arm]['brier']),f(a['single_seed']['metrics'][arm]['bce']),
         f(b['single_seed']['metrics'][arm]['brier']),f(b['single_seed']['metrics'][arm]['bce'])]
        for arm in b['single_seed']['metrics']])
    p('BCE uses natural logarithms (nats). Lower is better. Corrected fitted values are p11=0.8987854251012146, p01=0.2669902912621359 and marginal=0.7134670487106017. The corrected report equals the public frozen JSON structurally, including six-decimal scores. The source and corrected generator streams, estimates and rounded reports also match for seeds 20260921 through 20261120.')
    table(['Arm','Public-generator 200-seed mean Brier ± CI half-width','Mean BCE ± CI half-width'],[
        [arm,f(v['brier']['mean'])+' ± '+f(v['brier']['ci_half_width']),f(v['bce']['mean'])+' ± '+f(v['bce']['ci_half_width'])]
        for arm,v in b['benchmark']['summary'].items()])
    heading('Phase 2.3 — reset mathematics')
    p('The stationary reset Brier derivation is correct. The stated reset BCE 0.6869 is numerically incorrect; the correct value is 0.6873426691572835, rounding to 0.6873. These are known-parameter expected losses at a fixed 29-transition horizon, not the exact finite-training expected scores and not promised one-seed outcomes.')
    pre('a = p01 = 0.28; b = p11 = 0.92; λ = b − a = 0.64\nπ = a / (1 − b + a) = 7/9; v = π(1 − π); H = 29\nC(q,p) = −q ln(p) − (1−q) ln(1−p)\nC_D = (1−π)C(π,a) + π C(π,b)\nD = logit(b) − logit(a); g_H = (1−λ^H) / [H(1−λ)]\n\nStationary reset Brier(t) = v[1 + λ² − 2λ^(t+1)]\nStationary reset BCE(t)   = C_D − v λ^t D\nMean reset Brier = v(1+λ²) − 2vλ²g_H = 0.23007232801466346\nMean reset BCE   = C_D − vλDg_H      = 0.6873426691572835')
    p('At t=1 the reset and stateful predictions coincide. Under stationary initialization, their reset score approaches the deranged score as the initial state loses predictive information. The source starts instead with μ0=0.5, so the stationary covariance and stationary table do not apply unchanged.')
    pre('For the actual source, μ_t = π + (μ0−π)λ^t, with μ0=0.5.\nr_j(t) = P(X_t=1 | X_0=j) = π + (j−π)λ^t\nq̄ = (1/H) Σ[t=1..H] μ_t = 0.760749295777340\n\nFor either loss L = Brier risk or BCE risk:\nReset(t)    = (1−μ0)L(r_0(t),a) + μ0 L(r_1(t),b)\nDeranged(t) = (1−μ_(t−1))L(μ_t,a) + μ_(t−1)L(μ_t,b)')
    stationary=[('stateful_markov',.102044444444,.348588026364),('memoryless_marginal',.172839506173,.529706199058),('reset_each_trial',.230072328015,.687342669157),('deranged_memory',.243634567901,.723227599847)]
    public=[(.105450140845,.356947519589),(.182009804752,.550214690053),(.290633329580,.805453724914),(.255457586401,.753858858110)]
    table(['Arm','Stationary Brier','Stationary BCE','Source μ0=.5 Brier','Source μ0=.5 BCE'],[
        [row[0],f(row[1]),f(row[2]),f(other[0]),f(other[1])] for row,other in zip(stationary,public)])
    p('Under the public source, reset is worse than deranged in these expected losses. Its mean one-step forecast stays at 0.6 because it keeps using a balanced initial state, while the target probability approaches 7/9. The fitted pooled marginal converges to q̄ rather than 7/9 as training trials increase at fixed horizon. Finite-sample estimation and smoothing can change risk in either direction for the deliberately misspecified controls. The independent verification enumerates short binary paths and donor/target path pairs, and checks the stationary formulas at every t=1,…,29.')
    link('Full mathematical derivation and limitations','review/math_notes.txt')
    heading('Phase 2.4 — what the uploaded code actually achieves')
    table(['Uploaded arm','Brier, supplied seed','BCE, supplied seed'],[
        [arm,f(v['brier']),f(v['bce'])] for arm,v in c['single_seed']['metrics'].items()])
    p('All five arms begin at the true Regime-A parameters (0.92,0.28,7/9). That is privileged oracle initialization, not a fitted Phase 2.3 learner. The original full engine beats the frozen Regime-A forecast on this changing environment, but the no-self-model arm beats the full engine. Original 200-seed scores and paired comparisons are also included in phase24_original_benchmark.json.')
    table(['Finding','Correction or remaining implication'],[
        ['Reinitialization retains state and recent transitions','Reset now clears all run-dependent state and history; reuse is equivalent to a fresh engine.'],
        ['Zero learning rate can still clip extreme parameters','No-update behavior now preserves a frozen transition estimate exactly. Default original predictions are unchanged.'],
        ['Missing simple baselines and advertised shuffled control','Added plain two-parameter fixed EMA, fixed/adaptive memoryless predictors, and causal cross-stream scalar-state derangement.'],
        ['Ablation mislabeled parameter-matched baseline','The original fixed-rate arm retains memory and confidence arbitration; corrected label states this. No full model-capacity or optimization-budget matching is claimed.'],
        ['Oracle initialization mislabeled learned Phase 2.3','All corrected arms share estimates from independent 24×30 training trials. Phase 2.4 training uses stationary Regime-A initialization; it is explicitly an extension, not the exact public Phase 2.3 generator.'],
        ['Single-seed superiority asserted as unit test','Corrected tests check invariants and causal ordering; performance is reported across seeds and settings rather than enforced as universal ordering.'],
        ['Equation/code mismatch','Prose arousal weights 0.65/0.35 differ from code 0.70/0.30; prose linear learning rate differs from code 0.02+0.28 a^1.5. Corrected code preserves the supplied code dynamics and documents them.'],
        ['Memory schema differs from prose','Snapshots store p11, p01 and valence, not continuity; the valence field is not a retrieval score. This is parameter-snapshot memory, not demonstrated autobiographical memory.'],
        ['“Self-confidence” called calibration','An EMA of 1−Brier is a performance tracker. It does not separately measure probability calibration; prevalence and resolution affect it.'],
        ['Continuity and memory claims unvalidated','Cosine similarity is insensitive to proportional scale; no validated changepoint detector is established. A full memory bank has no eviction for novel contexts.'],
        ['No valence-only or continuity-only ablation','Results cannot isolate all proposed components. Arousal, memory gating and confidence interact; fixed thresholds remain heuristic.']])
    p('The original step_update computes the prediction before using the current outcome. Its use of that outcome for subsequent updates is valid online learning, not target leakage into the scored forecast. The corrected causal-prefix tests cover both the target stream and donor control. The corrected engine reproduces the original five arms’ normal-path predictions exactly when initialized identically; changes address reset, invalid inputs and boundary behavior.')
    heading('Phase 2.4 — stronger exploratory benchmark')
    m=d['metadata']
    p(f"Executed {m['n_primary']} primary seeds ({m['seeds_primary'][0]}–{m['seeds_primary'][1]}) and {m['n_holdout']} additional disjoint seeds ({m['seeds_holdout'][0]}–{m['seeds_holdout'][1]}), with {m['n_streams_per_seed']} independent streams per seed and {m['horizon']} scored transitions per stream. Tested stationary Regime A and alternating A/B blocks of 15, 45 and 90 transitions. Nine ordinary arms share data and learned initialization; two explicitly privileged oracle references bring the total to eleven. Hyperparameters were not tuned in this audit; the separate seed range is an exploratory robustness check, not a prospective preregistration.")
    p('The fixed EMA rate is the supplied 0.08, not a rate optimized on a validation set. The comparison therefore supports superiority to that specific simple baseline under these settings, not superiority over the EMA family or modern adaptive estimators. The original Regime B (0.18,0.82) is an alternating process with stationary probability 0.5; “sparse” is a misleading description.')
    table(['Arm','45-step switching: primary Brier','Primary BCE','Additional-seed Brier','Additional-seed BCE'],[
        [arm,f(v['brier']['mean']),f(v['bce']['mean']),
         f(d['summary']['holdout']['periodic_45']['metrics'][arm]['brier']['mean']),
         f(d['summary']['holdout']['periodic_45']['metrics'][arm]['bce']['mean'])]
        for arm,v in d['summary']['primary']['periodic_45']['metrics'].items()])
    rows=[]
    for split in ('primary','holdout'):
        for scenario,data in d['summary'][split].items():
            pair=data['paired_full_minus_comparator']['plain_fixed_ema']
            ci=pair['brier']['normal_mc_ci95'];ce=pair['bce']['normal_mc_ci95']
            rows.append([split,scenario,f(pair['brier']['mean'])+' ['+f(ci[0])+', '+f(ci[1])+']',
                         f(pair['bce']['mean'])+' ['+f(ce[0])+', '+f(ce[1])+']'])
    table(['Seeds','Environment','Full − plain EMA Brier [95% mean CI]','Full − plain EMA BCE [95% mean CI]'],rows)
    p('Negative differences favor the full engine. It improves over the specified plain EMA in all three switching settings, but degrades performance in stationary A. Removing the self-model lowers mean Brier and BCE in every setting in both seed ranges; for some close comparisons the uncertainty interval still includes zero. Memory removal increases mean loss in these settings, especially 45-step switching, supporting a conditional predictive contribution from that mechanism. The full combination is not the best tested variant.')
    p('The scalar-state derangement also lowers average losses relative to the full engine in every tested setting. This does not establish a benefit from properly aligned scalar states. The control copies only valence, arousal, confidence and continuity from another full-engine replica before any current-step outcome is revealed; fast/slow estimates, snapshots and recent history remain with the target. Donor streams have independent observations but the same regime schedule, so the donor can retain relevant regime information. It is not an unconditional independence null or a shuffle of all engine memory.')
    heading('How to interpret the uncertainty')
    p('Intervals are approximate normal Monte Carlo confidence intervals for a mean over simulated runs: mean ± 1.96×SD/√N. For Phase 2.4, first average the two streams within each seed, then use seeds as the replicate unit. Paired comparisons subtract scores within the same seed; transitions and shared-donor streams are not treated as independent samples. Scenarios share common random numbers and are not independent replications of one another. These intervals are not individual-run prediction intervals, human-population inference, or multiplicity-adjusted confirmatory tests. Raw per-seed and per-stream outputs are supplied for alternative analyses.')
    heading('Scientific scope and remaining work')
    p('The terms valence, arousal, self-model and continuity label numerical controller variables. Episodic memory here means stored transition-parameter snapshots. Neither predictive success nor model agreement establishes feelings, qualia, consciousness, or the MQGT-SCF fields Φc and E. This audit contains synthetic binary streams only, no human subjects, sensor measurements or phenomenological ground truth.')
    p('A stronger next forecasting study would select baseline learning rates on independent validation data, add discounted-count/sliding-window and change-detection or latent-regime baselines, score recovery around changes, test randomized or asynchronous regime durations and novel regimes, instrument snapshot retrieval, and include separate valence/continuity ablations. These investigations are not represented as completed by this report. The included current-regime oracle already supplies a known-parameter reference with privileged access to the true regime; the frozen Regime-A oracle makes the uploaded baseline explicit. Neither oracle is an information-matched learned model.')
    link('What the findings mean and the next falsifiable experiments','INTERPRETATION.txt')
    heading('Reproduce and inspect')
    pre('python3 -B run_all.py\n\n# Individual corrected scripts:\npython3 -B corrected/phase2_3_protocol_reproduction.py --seeds 200 \\\n  --output results/phase23_corrected_benchmark.json\npython3 -B corrected/phase2_4_functional_state_engine.py --seeds 200 \\\n  --holdout-seeds 100 --output results/phase24_corrected_benchmark.json\npython3 -B -m unittest discover -s corrected -p "test_*.py" -v\npython3 -B review/math_verify.py')
    p('Run from the extracted bundle directory with Python 3.10 or newer. The audit was actually executed on the runtime listed above; identical cross-version floating-point bytes are not promised. The runner needs no network access, reuses the pinned reference snapshot, fails on a nonzero command exit, and verifies original/reference integrity before and after. Output files in results/ are refreshed. Rebuilding this presentation is optional: python3 build_report.py.')
    for label,url in [('Corrected Phase 2.3 script','corrected/phase2_3_protocol_reproduction.py'),('Corrected Phase 2.4 script','corrected/phase2_4_functional_state_engine.py'),('One-command reproduction runner','run_all.py'),('Execution manifest','results/execution_manifest.json'),('Original Phase 2.3 actual console output','results/original_phase23.txt'),('Original Phase 2.4 actual console output','results/original_phase24.txt'),('Corrected test output','results/corrected_tests.txt'),('Corrected Phase 2.3 data','results/phase23_corrected_benchmark.json'),('Corrected Phase 2.4 raw results and paired intervals','results/phase24_corrected_benchmark.json'),('Corrected Phase 2.4 readable output','results/phase24_corrected_console.txt'),('Independent mathematics JSON','results/math_verification.json'),('Detailed Phase 2.4 audit','review/phase24_notes.txt')]:link(label,url)
    css='''body{margin:0;background:#f5f6f8;color:#172333;font:16px/1.6 system-ui,sans-serif}main{max-width:1160px;margin:40px auto;padding:44px;background:white;border-top:7px solid #315f75;box-shadow:0 4px 25px #1723330a}h1{font-size:34px;line-height:1.2;margin-top:0}h2{font-size:23px;margin-top:42px;border-bottom:1px solid #dce2e8;padding-bottom:9px}p{max-width:1050px}a{color:#175f88}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th{background:#223b4b;color:white;text-align:left}td,th{padding:10px 13px;border-bottom:1px solid #dce2e8;vertical-align:top}tbody tr:nth-child(even){background:#f3f7f9}.table-wrap{overflow-x:auto}pre{background:#eef3f6;padding:20px;white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}td{overflow-wrap:anywhere} @media(max-width:700px){main{margin:0;padding:20px}h1{font-size:27px}}@media print{body{background:white}main{margin:0;padding:0;box-shadow:none}a{color:inherit}tr{break-inside:avoid}h2{break-after:avoid}}'''
    (ROOT/'verification_report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ZoraASI executed verification</title><style>'+css+'</style></head><body><main>'+''.join(parts)+'</main></body></html>')
    (ROOT/'verification_report.txt').write_text('\n'.join(plain)+'\n')
    for original,fixed,name in [('original/phase2_3_protocol_reproduction.py','corrected/phase2_3_protocol_reproduction.py','phase23_corrections.diff'),('original/synthetic_qualia_engine_v1.py','corrected/phase2_4_functional_state_engine.py','phase24_corrections.diff')]:
        delta=''.join(difflib.unified_diff((ROOT/original).read_text().splitlines(True),(ROOT/fixed).read_text().splitlines(True),fromfile=original,tofile=fixed))
        (ROOT/'review'/name).write_text(delta)
    print('Wrote HTML/text verification report and correction diffs.')

if __name__=='__main__':main()
