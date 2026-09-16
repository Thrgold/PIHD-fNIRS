# Verified result summary and scope

Values below were recomputed from the packaged JSON/NumPy outputs. These checks
establish internal consistency and correspondence with the analysis code; they do
not replace an independent rerun from the source data.

## Dataset and evaluation

- 48 participants and 480 trials from the preprocessed SEFMID data.
- Stochastic PIHD analyses use seeds 42, 123, and 999.
- Classification uses pooled leave-one-subject-out (LOSO) AUC: all held-out
  predictions are concatenated before computing one AUC.
- Chance tests use 1,000 global trial-label permutations while preserving LOSO
  groups and folds. The smallest attainable corrected-count p-value is 1/1001.
- ME and MI activation endpoints are participant-level task-minus-preceding-rest
  Cohen's dz, with Bonferroni correction across 18 method-by-task tests.

## Main findings

PIHD pooled LOSO AUC was 0.631 +/- 0.004 (mean +/- sample SD across seeds), and
each seed gave permutation p=0.001. Deterministic baseline AUCs were Raw HbO
0.543, bandpass 0.557, GLM 0.514, ICA 0.593, PCA 0.574, Wavelet 0.609, SSA 0.575,
and EMD 0.533. Baseline seed-specific permutation p-values vary because the
permutation draw changes; their observed predictions are deterministic.

PIHD exceeded Wavelet numerically by Delta AUC = 0.022 +/- 0.004, but the
participant-level prediction-swap tests were nonsignificant (p=0.253--0.399), and
all seed-specific participant-cluster bootstrap 95% CIs crossed zero. The evidence
supports the highest observed point estimate, not statistical superiority.

PIHD task-minus-rest effects were ME dz = 1.02 +/- 0.11 and MI dz = 1.16 +/- 0.19;
both were Bonferroni-significant in every seed. None of the eight baselines showed
a significant positive contrast under this analysis. These directional contrasts
are not direct neural measurements and do not establish neural specificity.

## Controls and sensitivity analyses

- Removing the ODE residual reduced pooled LOSO AUC to 0.556 +/- 0.010; removing
  Stage 1 reduced it to 0.605 +/- 0.005.
- Gradient-matched curvature smoothing achieved 0.531 +/- 0.034. The three-seed
  mean PIHD-minus-smoothing difference was 0.099, participant-cluster bootstrap
  95% CI [0.039, 0.160], paired-swap p=0.0034.
- Correct task timing had the highest mean AUC (0.631) among the tested shifts,
  whose means ranged from 0.561 to 0.604. Positive contrasts persisted after
  shifting and were often larger, so the contrast was not timing-specific.
- Across lambda_ode in {0.25, 0.5, 1, 2, 4}, lambda_ode=1 had the highest mean
  AUC (0.631). Lambda_ode=2 increased MI dz to 1.45 +/- 0.14 while lowering AUC
  to 0.621 +/- 0.004. The chosen weight balances the evaluated endpoints rather
  than identifying a unique physiological optimum.
- Full offline per-trial optimization on an RTX 4060 Laptop GPU took 12.11 +/-
  0.87 s over 30 measured trials. This is optimization time, not trained-network
  forward inference and not evidence of online operation.

## Interpretation boundary

The results support a paradigm-informed hemodynamic representation constrained by
the implemented simplified Balloon--Windkessel residuals on one dataset. They do
not prove recovery of a unique neural source, causal neural specificity,
generalization to another dataset, online readiness, or superiority over Wavelet.
The learned tau and epsilon are model parameters, not independently validated
physiological measurements.
