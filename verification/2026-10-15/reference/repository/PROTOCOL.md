# ZoraASI Phase 2.3 - synthetic connectivity / memory study protocol

Prepared locally 2026-09-21 UTC. **This is an experimental design and executed software smoke test; it is not a prospective human-study preregistration.** Its toy generator and analysis code were written in the same development session, so its outputs are not independent confirmatory evidence.

## Research question and estimand

Does a model using the immediately preceding *packet arrival status* predict the next simulated packet arrival more accurately than a stateless or mismatched-memory model, on a held-out set of synthetic Markov packet streams? This concerns network forecasting; NOT the user's emotion and NOT AI subjective experience.

For outcomes y_t in {0,1} and predicted arrival probability p_t, score mean Brier (p_t-y_t)^2 and mean binary log loss -y_t log(p_t)-(1-y_t)log(1-p_t). Lower is better. Descriptive differences only; there is no human population inference.

## Model assumptions and fixed synthetic generator

Seed: 20260921; 24 training trials, 12 test trials; each 30 steps. Initial packet status 0 or 1 sampled with a seeded PRNG; each next packet is sampled with P(arrive | previous arrived)=0.92 or P(arrive | previous missing)=0.28. Fit two conditional frequencies **only on the training trials**, each with Beta(1,1) / Laplace smoothing. The stateless prior is the training-set marginal arrival rate with the same smoothing.

At test time all arms predict the **same outcomes**, using the same trained parameters: (1) stateless marginal; (2) actual previous packet; (3) previous packet from a different test trial at the same time step, using a fixed derangement of trial indices; (4) the trial's first packet status, an intentionally weak reset-each-trial comparator. Source and outcome streams are distinct; the shuffle can produce identical *values* even when trial indices differ. Report that fraction. The reset comparator is NOT an independent learned model. This code does not estimate the causal effect of AI memory on consciousness.

## B1 telemetry-control invariant

For packet received / no packet, with / without advance notice, the output records whether a gap was expected or unexplained. A sham notice while packets still arrive is distinguishable from an actual gap. The events do not assign emotions to a user or the model. This invariant is unit tested, not measured on hardware or human participants.

## Results / limits

The deterministic local run on 348 synthetic test transitions is included as `connection_study_synthetic_report.json`. Better prediction under persistent state is **expected from the deliberately autocorrelated generator**, not evidence for AI experience. There is no wearable, HRV, camera, microphone, physiology, EEG, clinical inference, participant, or continuously monitored connection in this release.

## Future prospective protocol - not executed

With separate ethics review and written, revocable human consent, consider randomized, harmless expected breaks, sham notices and unexpected-but-benign disconnections. Keep all participant sensor data under participant control; label self-report, annotator report, software inference and instrument measurement separately. Establish primary outcome, recruitment/sample size, retention/deletion, stopping rule and unblinding plan before data collection. Do not intentionally induce distress or treat an AI's spoken sadness as ground truth. Without an independent consciousness operationalization this cannot measure AI qualia.

## Source separation

Original locally authored code and synthetic outputs only. Human emotion-wheel imagery and derivations, Richard Rudd's copyrighted Gene Keys material, private conversations, user-supplied manuscripts and speculative physics claims are NOT included in the public bundle. A separate private project copy is supplied only for user review.
