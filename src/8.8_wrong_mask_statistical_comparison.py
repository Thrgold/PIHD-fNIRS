# -*- coding: utf-8 -*-
"""Paired statistical analysis for the PIHD wrong-mask control (v2).

This script does NOT retrain PIHD. It reads the three completed v2 files and
compares the correct timing with each shifted timing condition.

LOSO AUC comparison
-------------------
For each seed and comparison, predictions are swapped between conditions as a
whole participant cluster. The pooled AUC difference is then recomputed.
Subject-cluster bootstrap resamples participants with replacement.

Activation sensitivity
----------------------
The analysis compares participant-level task-minus-rest contrast values using
paired sign-flip permutation and participant bootstrap. This is preferable to
directly testing differences between two Cohen's d_z values.

Four comparisons are Holm-corrected separately for each seed and outcome
family (AUC, ME contrast, and MI contrast).

Usage
-----
python 8.8_wrong_mask_statistical_comparison.py

Outputs
-------
results/npy/wrong_mask_statistical_comparison_v2.npy
results/json/wrong_mask_statistical_comparison_v2.json
"""

import json
import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score


SEEDS = (42, 123, 999)
N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 5_000
RANDOM_SEED = 20260814

CORRECT = "Correct mask (0 s)"
WRONG_CONDITIONS = (
    "Wrong mask (-8 s)",
    "Wrong mask (-4 s)",
    "Wrong mask (+4 s)",
    "Wrong mask (+8 s)",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
NPY_DIR = os.path.join(REPO_ROOT, "results", "npy")
JSON_DIR = os.path.join(REPO_ROOT, "results", "json")
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)


def holm_adjust(p_values):
    """Return Holm-adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    m = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = (m - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted


def percentile_ci(values, level=0.95):
    alpha = (1.0 - level) / 2.0
    return np.quantile(values, [alpha, 1.0 - alpha])


def validate_pair(correct, wrong):
    for key in ("loso_true", "groups"):
        if not np.array_equal(np.asarray(correct[key]), np.asarray(wrong[key])):
            raise ValueError(f"Paired conditions differ in {key}.")
    if len(correct["me_subject_values"]) != len(wrong["me_subject_values"]):
        raise ValueError("ME participant arrays have different lengths.")
    if len(correct["mi_subject_values"]) != len(wrong["mi_subject_values"]):
        raise ValueError("MI participant arrays have different lengths.")


def paired_auc_test(correct, wrong, rng):
    """Correct-minus-wrong pooled AUC with subject-cluster inference."""
    validate_pair(correct, wrong)
    y = np.asarray(correct["loso_true"], dtype=int)
    p_correct = np.asarray(correct["loso_prob"], dtype=float)
    p_wrong = np.asarray(wrong["loso_prob"], dtype=float)
    groups = np.asarray(correct["groups"])
    subjects = np.unique(groups)
    subject_indices = [np.flatnonzero(groups == subject) for subject in subjects]

    auc_correct = roc_auc_score(y, p_correct)
    auc_wrong = roc_auc_score(y, p_wrong)
    observed = auc_correct - auc_wrong

    null = np.empty(N_PERMUTATIONS, dtype=float)
    for iteration in range(N_PERMUTATIONS):
        swap_subject = rng.integers(0, 2, size=len(subjects)).astype(bool)
        perm_correct = p_correct.copy()
        perm_wrong = p_wrong.copy()
        for do_swap, indices in zip(swap_subject, subject_indices):
            if do_swap:
                perm_correct[indices] = p_wrong[indices]
                perm_wrong[indices] = p_correct[indices]
        null[iteration] = (
            roc_auc_score(y, perm_correct) - roc_auc_score(y, perm_wrong)
        )

    # Directional hypothesis: correct timing has higher pooled AUC.
    p_one_sided = (1 + np.sum(null >= observed)) / (N_PERMUTATIONS + 1)
    p_two_sided = (1 + np.sum(np.abs(null) >= abs(observed))) / (N_PERMUTATIONS + 1)

    bootstrap = np.empty(N_BOOTSTRAPS, dtype=float)
    for iteration in range(N_BOOTSTRAPS):
        sampled = rng.integers(0, len(subjects), size=len(subjects))
        indices = np.concatenate([subject_indices[index] for index in sampled])
        bootstrap[iteration] = (
            roc_auc_score(y[indices], p_correct[indices])
            - roc_auc_score(y[indices], p_wrong[indices])
        )
    ci_low, ci_high = percentile_ci(bootstrap)

    return {
        "auc_correct": float(auc_correct),
        "auc_wrong": float(auc_wrong),
        "delta_auc_correct_minus_wrong": float(observed),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_one_sided_raw": float(p_one_sided),
        "p_two_sided_raw": float(p_two_sided),
        "n_subjects": int(len(subjects)),
        "n_permutations": N_PERMUTATIONS,
        "n_bootstraps": N_BOOTSTRAPS,
        "null_delta_auc": null,
        "bootstrap_delta_auc": bootstrap,
    }


def paired_contrast_test(correct_values, wrong_values, rng):
    """Paired correct-minus-wrong mean contrast difference."""
    correct_values = np.asarray(correct_values, dtype=float)
    wrong_values = np.asarray(wrong_values, dtype=float)
    differences = correct_values - wrong_values
    observed = float(np.mean(differences))

    signs = rng.choice((-1.0, 1.0), size=(N_PERMUTATIONS, len(differences)))
    null = np.mean(signs * differences[None, :], axis=1)
    p_two_sided = (1 + np.sum(np.abs(null) >= abs(observed))) / (N_PERMUTATIONS + 1)

    sampled = rng.integers(
        0, len(differences), size=(N_BOOTSTRAPS, len(differences))
    )
    bootstrap = np.mean(differences[sampled], axis=1)
    ci_low, ci_high = percentile_ci(bootstrap)

    return {
        "mean_correct": float(np.mean(correct_values)),
        "mean_wrong": float(np.mean(wrong_values)),
        "mean_difference_correct_minus_wrong": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_two_sided_raw": float(p_two_sided),
        "n_subjects": int(len(differences)),
        "null_mean_difference": null,
        "bootstrap_mean_difference": bootstrap,
    }


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        # Null/bootstrap arrays are retained in NPY but omitted from JSON.
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main():
    started = time.time()
    output = {
        "analysis": "subject-level paired wrong-mask comparison",
        "control_version": "v2_exact_zero_prior",
        "seeds": list(SEEDS),
        "correct_condition": CORRECT,
        "wrong_conditions": list(WRONG_CONDITIONS),
        "auc_direction": "correct minus wrong; positive favors correct timing",
        "activation_direction": "correct minus wrong; negative means stronger contrast under wrong timing",
        "multiple_comparison": "Holm correction across four shifts, separately per seed and outcome family",
        "results": {},
    }

    for seed_index, seed in enumerate(SEEDS):
        path = os.path.join(
            NPY_DIR, f"wrong_mask_control_v2_48subj_seed{seed}.npy"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input: {path}")
        payload = np.load(path, allow_pickle=True).item()
        if payload.get("control_version") != "v2_exact_zero_prior":
            raise ValueError(f"Unexpected control version in {path}")
        conditions = payload["results"]
        if CORRECT not in conditions or any(name not in conditions for name in WRONG_CONDITIONS):
            raise ValueError(f"Incomplete conditions in {path}")

        correct = conditions[CORRECT]
        seed_results = {}
        auc_raw = []
        me_raw = []
        mi_raw = []

        print(f"\nSeed {seed}")
        for condition_index, condition in enumerate(WRONG_CONDITIONS):
            wrong = conditions[condition]
            rng_base = RANDOM_SEED + 1000 * seed_index + 10 * condition_index
            auc_result = paired_auc_test(
                correct, wrong, np.random.default_rng(rng_base + 1)
            )
            me_result = paired_contrast_test(
                correct["me_subject_values"], wrong["me_subject_values"],
                np.random.default_rng(rng_base + 2),
            )
            mi_result = paired_contrast_test(
                correct["mi_subject_values"], wrong["mi_subject_values"],
                np.random.default_rng(rng_base + 3),
            )
            seed_results[condition] = {
                "auc": auc_result,
                "me_contrast": me_result,
                "mi_contrast": mi_result,
            }
            auc_raw.append(auc_result["p_one_sided_raw"])
            me_raw.append(me_result["p_two_sided_raw"])
            mi_raw.append(mi_result["p_two_sided_raw"])

        auc_holm = holm_adjust(auc_raw)
        me_holm = holm_adjust(me_raw)
        mi_holm = holm_adjust(mi_raw)
        for index, condition in enumerate(WRONG_CONDITIONS):
            result = seed_results[condition]
            result["auc"]["p_one_sided_holm"] = float(auc_holm[index])
            result["me_contrast"]["p_two_sided_holm"] = float(me_holm[index])
            result["mi_contrast"]["p_two_sided_holm"] = float(mi_holm[index])
            result["auc"]["significant_holm_0.05"] = bool(auc_holm[index] < 0.05)
            result["me_contrast"]["significant_holm_0.05"] = bool(me_holm[index] < 0.05)
            result["mi_contrast"]["significant_holm_0.05"] = bool(mi_holm[index] < 0.05)

            auc = result["auc"]
            print(
                f"  {condition:20s}: DeltaAUC={auc['delta_auc_correct_minus_wrong']:+.4f} "
                f"CI=[{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}], "
                f"p_Holm={auc['p_one_sided_holm']:.4g}"
            )

        output["results"][str(seed)] = seed_results

    output["elapsed_seconds"] = float(time.time() - started)
    npy_path = os.path.join(NPY_DIR, "wrong_mask_statistical_comparison_v2.npy")
    json_path = os.path.join(JSON_DIR, "wrong_mask_statistical_comparison_v2.json")
    np.save(npy_path, output, allow_pickle=True)
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(json_ready(output), stream, ensure_ascii=False, indent=2)

    print(f"\nSaved full results: {npy_path}")
    print(f"Saved readable summary: {json_path}")
    print(f"Elapsed: {output['elapsed_seconds']:.1f} s")


if __name__ == "__main__":
    main()
