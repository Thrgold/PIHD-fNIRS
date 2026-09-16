# -*- coding: utf-8 -*-
"""Across-seed summary for the PIHD vs matched-smoothness control.

This script performs no decomposition and no classifier fitting.  It reads the
three completed Step 6.6-v2 files and treats participants, not random seeds, as
the resampling unit.  The primary statistic is the mean of the three seed-wise
pooled-LOSO AUC differences.  The same subject-level swap is applied to all
three seeds in each permutation, preserving cross-seed dependence.

Usage
-----
python 6.7_smoothness_control_v2_summary.py

Outputs
-------
results/npy/smoothness_control_v2_summary.npy
results/json/smoothness_control_v2_summary.json
"""

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

# Prevent Windows consoles from failing when the project path contains Chinese text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SEEDS = (42, 123, 999)
N_PAIRED_PERM = 10000
N_CLUSTER_BOOT = 10000
RNG_SEED_PERM = 672024
RNG_SEED_BOOT = 672025

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
NPY_DIR = os.path.join(REPO_ROOT, "results", "npy")
JSON_DIR = os.path.join(REPO_ROOT, "results", "json")
os.makedirs(JSON_DIR, exist_ok=True)

OUTPUT_NPY = os.path.join(NPY_DIR, "smoothness_control_v2_summary.npy")
OUTPUT_JSON = os.path.join(JSON_DIR, "smoothness_control_v2_summary.json")


def fast_auc(labels, scores):
    """Mann--Whitney form of ROC AUC; supports tied scores."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_positive = int(np.sum(labels == 1))
    n_negative = int(np.sum(labels == 0))
    if n_positive == 0 or n_negative == 0:
        return np.nan
    ranks = rankdata(scores, method="average")
    positive_rank_sum = np.sum(ranks[labels == 1])
    return float(
        (positive_rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def mean_seed_delta(labels, full_probabilities, control_probabilities, index):
    differences = []
    indexed_labels = labels[index]
    for seed_index in range(len(SEEDS)):
        differences.append(
            fast_auc(indexed_labels, full_probabilities[seed_index, index])
            - fast_auc(indexed_labels, control_probabilities[seed_index, index])
        )
    return float(np.mean(differences))


payloads = []
for seed in SEEDS:
    path = os.path.join(
        NPY_DIR, f"smoothness_control_v2_48subj_seed{seed}.npy"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing v2 result: {path}")
    payload = np.load(path, allow_pickle=True).item()
    if payload.get("seed") != seed:
        raise ValueError(f"Seed metadata mismatch: {path}")
    if payload.get("n_subjects") != 48 or payload.get("n_trials") != 480:
        raise ValueError(f"Incomplete result: {path}")
    matching = payload["weight_matching"]
    if not np.all(np.asarray(matching["matched_flags"], dtype=bool)):
        raise ValueError(f"Unmatched trial found: {path}")
    relative_error = np.asarray(
        matching["matched_gradient_relative_errors"], dtype=float
    )
    if not np.all(np.isfinite(relative_error)) or np.max(relative_error) > 1e-5:
        raise ValueError(f"Invalid gradient matching diagnostics: {path}")
    payloads.append(payload)

labels = np.asarray(payloads[0]["labels"], dtype=int)
groups = np.asarray(payloads[0]["groups"])
for payload in payloads[1:]:
    if not np.array_equal(labels, np.asarray(payload["labels"], dtype=int)):
        raise ValueError("Label order differs across seeds")
    if not np.array_equal(groups, np.asarray(payload["groups"])):
        raise ValueError("Group order differs across seeds")

full_probabilities = np.asarray(
    [payload["full_pihd_reference"]["loso_prob"] for payload in payloads],
    dtype=float,
)
control_probabilities = np.asarray(
    [payload["classification"]["loso_prob"] for payload in payloads],
    dtype=float,
)
all_index = np.arange(len(labels))
unique_subjects = np.asarray(sorted(np.unique(groups)))
subject_index = {
    subject: np.flatnonzero(groups == subject) for subject in unique_subjects
}

full_auc_by_seed = np.asarray(
    [fast_auc(labels, probability) for probability in full_probabilities]
)
control_auc_by_seed = np.asarray(
    [fast_auc(labels, probability) for probability in control_probabilities]
)
delta_auc_by_seed = full_auc_by_seed - control_auc_by_seed
observed_mean_delta = float(np.mean(delta_auc_by_seed))

# Paired subject-level prediction swaps.  A subject is swapped in all seeds
# simultaneously so seed replicates are not treated as independent samples.
rng_perm = np.random.RandomState(RNG_SEED_PERM)
null_mean_delta = np.empty(N_PAIRED_PERM, dtype=float)
for iteration in range(N_PAIRED_PERM):
    permuted_full = full_probabilities.copy()
    permuted_control = control_probabilities.copy()
    swap_subjects = unique_subjects[
        rng_perm.rand(len(unique_subjects)) < 0.5
    ]
    for subject in swap_subjects:
        index = subject_index[subject]
        temporary = permuted_full[:, index].copy()
        permuted_full[:, index] = permuted_control[:, index]
        permuted_control[:, index] = temporary
    null_mean_delta[iteration] = mean_seed_delta(
        labels, permuted_full, permuted_control, all_index
    )

paired_p_two_sided = float(
    (1 + np.sum(np.abs(null_mean_delta) >= abs(observed_mean_delta)))
    / (N_PAIRED_PERM + 1)
)

# Participant-cluster bootstrap, again retaining all three predictions for each
# sampled participant.  Duplicate subject blocks are valid bootstrap clusters.
rng_boot = np.random.RandomState(RNG_SEED_BOOT)
bootstrap_mean_delta = np.empty(N_CLUSTER_BOOT, dtype=float)
for iteration in range(N_CLUSTER_BOOT):
    sampled_subjects = rng_boot.choice(
        unique_subjects, size=len(unique_subjects), replace=True
    )
    sampled_index = np.concatenate(
        [subject_index[subject] for subject in sampled_subjects]
    )
    bootstrap_mean_delta[iteration] = mean_seed_delta(
        labels, full_probabilities, control_probabilities, sampled_index
    )

valid_bootstrap = bootstrap_mean_delta[np.isfinite(bootstrap_mean_delta)]
ci95 = np.percentile(valid_bootstrap, [2.5, 97.5])

me_dz = np.asarray(
    [payload["activation"]["me_dz"] for payload in payloads], dtype=float
)
mi_dz = np.asarray(
    [payload["activation"]["mi_dz"] for payload in payloads], dtype=float
)
lambda_values = np.concatenate(
    [payload["weight_matching"]["lambda_values"] for payload in payloads]
).astype(float)
relative_errors = np.concatenate(
    [
        payload["weight_matching"]["matched_gradient_relative_errors"]
        for payload in payloads
    ]
).astype(float)

summary = {
    "seeds": np.asarray(SEEDS, dtype=int),
    "n_subjects": len(unique_subjects),
    "n_trials": len(labels),
    "primary_statistic": "mean of seed-wise pooled-LOSO AUC differences",
    "full_pihd_auc_by_seed": full_auc_by_seed,
    "smoothness_auc_by_seed": control_auc_by_seed,
    "delta_auc_by_seed": delta_auc_by_seed,
    "full_pihd_auc_mean": float(np.mean(full_auc_by_seed)),
    "full_pihd_auc_sd": float(np.std(full_auc_by_seed, ddof=1)),
    "smoothness_auc_mean": float(np.mean(control_auc_by_seed)),
    "smoothness_auc_sd": float(np.std(control_auc_by_seed, ddof=1)),
    "mean_delta_auc": observed_mean_delta,
    "mean_delta_auc_sd_across_seeds": float(
        np.std(delta_auc_by_seed, ddof=1)
    ),
    "participant_cluster_bootstrap_ci95": ci95,
    "participant_paired_swap_p_two_sided": paired_p_two_sided,
    "paired_swap_null_mean_delta": null_mean_delta,
    "cluster_bootstrap_mean_delta": bootstrap_mean_delta,
    "smoothness_me_dz_by_seed": me_dz,
    "smoothness_mi_dz_by_seed": mi_dz,
    "smoothness_me_dz_mean": float(np.mean(me_dz)),
    "smoothness_me_dz_sd": float(np.std(me_dz, ddof=1)),
    "smoothness_mi_dz_mean": float(np.mean(mi_dz)),
    "smoothness_mi_dz_sd": float(np.std(mi_dz, ddof=1)),
    "matching_diagnostics": {
        "n_trials_all_seeds": len(lambda_values),
        "all_trials_matched": True,
        "coefficient_clipping": False,
        "lambda_median": float(np.median(lambda_values)),
        "lambda_iqr": np.percentile(lambda_values, [25, 75]),
        "lambda_range": np.asarray(
            [np.min(lambda_values), np.max(lambda_values)]
        ),
        "maximum_relative_error": float(np.max(relative_errors)),
    },
}
np.save(OUTPUT_NPY, summary, allow_pickle=True)

json_summary = {
    "seeds": list(SEEDS),
    "n_subjects": len(unique_subjects),
    "n_trials": len(labels),
    "primary_statistic": summary["primary_statistic"],
    "full_pihd_auc": {
        "by_seed": full_auc_by_seed.tolist(),
        "mean": summary["full_pihd_auc_mean"],
        "sd": summary["full_pihd_auc_sd"],
    },
    "smoothness_auc": {
        "by_seed": control_auc_by_seed.tolist(),
        "mean": summary["smoothness_auc_mean"],
        "sd": summary["smoothness_auc_sd"],
    },
    "full_minus_smoothness": {
        "by_seed": delta_auc_by_seed.tolist(),
        "mean": observed_mean_delta,
        "sd_across_seeds": summary["mean_delta_auc_sd_across_seeds"],
        "participant_cluster_bootstrap_ci95": ci95.tolist(),
        "participant_paired_swap_p_two_sided": paired_p_two_sided,
    },
    "smoothness_activation": {
        "me_dz_by_seed": me_dz.tolist(),
        "me_dz_mean": summary["smoothness_me_dz_mean"],
        "me_dz_sd": summary["smoothness_me_dz_sd"],
        "mi_dz_by_seed": mi_dz.tolist(),
        "mi_dz_mean": summary["smoothness_mi_dz_mean"],
        "mi_dz_sd": summary["smoothness_mi_dz_sd"],
    },
    "matching_diagnostics": {
        "n_trials_all_seeds": len(lambda_values),
        "all_trials_matched": True,
        "coefficient_clipping": False,
        "lambda_median": summary["matching_diagnostics"]["lambda_median"],
        "lambda_iqr": summary["matching_diagnostics"]["lambda_iqr"].tolist(),
        "lambda_range": summary["matching_diagnostics"]["lambda_range"].tolist(),
        "maximum_relative_error": summary["matching_diagnostics"]["maximum_relative_error"],
    },
}
with open(OUTPUT_JSON, "w", encoding="utf-8") as stream:
    json.dump(json_summary, stream, ensure_ascii=False, indent=2)

print("=" * 76)
print("PIHD vs Gradient-Strength-Matched Smoothness Control (3-seed summary)")
print("=" * 76)
print(
    f"Full PIHD AUC:       {summary['full_pihd_auc_mean']:.3f} "
    f"+/- {summary['full_pihd_auc_sd']:.3f}"
)
print(
    f"Smoothness AUC:      {summary['smoothness_auc_mean']:.3f} "
    f"+/- {summary['smoothness_auc_sd']:.3f}"
)
print(
    f"Mean Delta AUC:      {observed_mean_delta:+.3f} "
    f"+/- {summary['mean_delta_auc_sd_across_seeds']:.3f}"
)
print(f"Cluster CI95:        [{ci95[0]:+.3f}, {ci95[1]:+.3f}]")
print(f"Paired swap p:       {paired_p_two_sided:.4f} (two-sided)")
print(
    f"Smoothness ME d_z:   {summary['smoothness_me_dz_mean']:+.3f} "
    f"+/- {summary['smoothness_me_dz_sd']:.3f}"
)
print(
    f"Smoothness MI d_z:   {summary['smoothness_mi_dz_mean']:+.3f} "
    f"+/- {summary['smoothness_mi_dz_sd']:.3f}"
)
print(
    f"Max matching error:  "
    f"{summary['matching_diagnostics']['maximum_relative_error']:.3e}"
)
print("=" * 76)
print(f"NPY saved:  {OUTPUT_NPY}")
print(f"JSON saved: {OUTPUT_JSON}")
