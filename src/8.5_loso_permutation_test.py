# -*- coding: utf-8 -*-
"""Step 8.5: LOSO pooled-AUC permutation testing.

Reads the feature matrix saved by 7_loso_classification_9methods_48subj.py.
No fNIRS preprocessing or PIHD decomposition is performed here.

Usage
-----
python 8.5_loso_permutation_test.py          # uses seed 42
python 8.5_loso_permutation_test.py 123

The input and output files are resolved relative to this script's parent
directory, e.g. <project>/results/... when this script is in <project>/scripts.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
N_PERM = 1000
N_BOOTSTRAP = 2000
N_COMPARE_PERM = 10000
METHODS = ("raw", "bandpass", "glm", "ica", "pca", "wavelet", "ssa", "emd", "pihd")
METHOD_LABELS = {
    "raw": "Raw HbO", "bandpass": "Bandpass", "glm": "GLM", "ica": "ICA",
    "pca": "PCA", "wavelet": "Wavelet", "ssa": "SSA", "emd": "EMD", "pihd": "PIHD",
}


def pooled_auc(y_true, scores):
    """Return ROC AUC, or NaN if it is undefined."""
    y_true = np.asarray(y_true)
    if np.unique(y_true).size != 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def run_loso(X, y, groups):
    """Fit scaler and classifier within each LOSO training set only."""
    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    true_parts, score_parts, group_parts = [], [], []
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X, y, groups):
        if np.unique(y[train_idx]).size != 2:
            return np.nan, None, None, None
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        classifier.fit(X_train, y[train_idx])
        true_parts.append(y[test_idx])
        score_parts.append(classifier.predict_proba(X_test)[:, 1])
        group_parts.append(groups[test_idx])

    y_all = np.concatenate(true_parts)
    score_all = np.concatenate(score_parts)
    group_all = np.concatenate(group_parts)
    return pooled_auc(y_all, score_all), y_all, score_all, group_all


def permutation_test(X, y, groups, observed_auc, rng, method_label):
    """Globally permute trial labels while retaining groups and LOSO folds."""
    null_auc = np.full(N_PERM, np.nan)
    started = time.time()
    for index in range(N_PERM):
        permuted_y = rng.permutation(y)
        null_auc[index] = run_loso(X, permuted_y, groups)[0]
        if (index + 1) % 100 == 0 or index == 0:
            print(f"  {method_label:<12} {index + 1:4d}/{N_PERM}  "
                  f"({time.time() - started:.1f}s)", flush=True)
    valid = null_auc[np.isfinite(null_auc)]
    p_value = (1 + np.sum(valid >= observed_auc)) / (valid.size + 1)
    return null_auc, float(p_value)


def cluster_bootstrap_delta(y, pihd_scores, wavelet_scores, groups, rng):
    """Percentile 95% CI for ΔAUC using subject (not trial) resampling."""
    subject_values = np.unique(groups)
    deltas = []
    # Resampling can very rarely create a one-class sample; redraw it.
    while len(deltas) < N_BOOTSTRAP:
        sampled = rng.choice(subject_values, len(subject_values), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == subject) for subject in sampled])
        delta = pooled_auc(y[indices], pihd_scores[indices]) - pooled_auc(y[indices], wavelet_scores[indices])
        if np.isfinite(delta):
            deltas.append(delta)
    deltas = np.asarray(deltas)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), deltas


def subject_swap_test(y, pihd_scores, wavelet_scores, groups, rng):
    """Two-sided paired prediction-swap test, swapping complete subjects."""
    observed = pooled_auc(y, pihd_scores) - pooled_auc(y, wavelet_scores)
    subject_values = np.unique(groups)
    null_delta = np.empty(N_COMPARE_PERM)
    for index in range(N_COMPARE_PERM):
        first, second = pihd_scores.copy(), wavelet_scores.copy()
        for subject, should_swap in zip(subject_values, rng.integers(0, 2, subject_values.size)):
            if should_swap:
                mask = groups == subject
                first[mask], second[mask] = second[mask].copy(), first[mask].copy()
        null_delta[index] = pooled_auc(y, first) - pooled_auc(y, second)
    p_value = (1 + np.sum(np.abs(null_delta) >= abs(observed))) / (N_COMPARE_PERM + 1)
    return float(observed), null_delta, float(p_value)


def significance(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def json_ready_method(observed, null_auc, p_value):
    return {
        "observed_pooled_auc": float(observed),
        "permutation_p": float(p_value), "significance": significance(p_value),
        "null_auc_mean": float(np.nanmean(null_auc)), "null_auc_std": float(np.nanstd(null_auc)),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    # Supports placing the script in the project root, scripts/, code/, or an
    # exported-output folder. The project's convention stores .npy results in
    # results/npy; retain results/ as a backward-compatible fallback.
    candidate_dirs = (script_dir, script_dir.parent)
    result_candidates = [directory / "results" / "npy" for directory in candidate_dirs]
    result_candidates += [directory / "results" for directory in candidate_dirs]
    results_dir = next((directory for directory in result_candidates if directory.exists()),
                       script_dir / "results" / "npy")
    input_path = results_dir / f"loso_classification_9methods_seed{SEED}.npy"
    output_npy = results_dir / f"loso_permutation_9methods_seed{SEED}.npy"
    output_json = results_dir / f"loso_permutation_9methods_seed{SEED}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}\nRun script 7 with seed {SEED} first.")

    data = np.load(input_path, allow_pickle=True).item()
    required = {"features", "labels", "groups", "subject_ids"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Input file is missing required key(s): {sorted(missing)}")
    features = data["features"]
    y = np.asarray(data["labels"], dtype=int)
    groups = np.asarray(data["groups"])
    if y.size != groups.size:
        raise ValueError("labels and groups must have the same number of trials.")
    absent = [method for method in METHODS if method not in features]
    if absent:
        raise KeyError(f"Feature data missing for: {absent}")

    print(f"LOSO permutation test | seed={SEED} | trials={y.size} | subjects={np.unique(groups).size}")
    observed, predictions, permutation_results, summary = {}, {}, {}, {}
    for method_index, method in enumerate(METHODS):
        X = np.asarray(features[method], dtype=float)
        if X.shape[0] != y.size:
            raise ValueError(f"{method}: features have {X.shape[0]} rows, expected {y.size}.")
        auc, y_out, probabilities, out_groups = run_loso(X, y, groups)
        observed[method] = auc
        predictions[method] = (y_out, probabilities, out_groups)
        print(f"Observed {METHOD_LABELS[method]:<12}: pooled AUC={auc:.4f}")
        null_auc, p_value = permutation_test(X, y, groups, auc,
                                              np.random.default_rng(SEED * 1000 + method_index),
                                              METHOD_LABELS[method])
        permutation_results[method] = {"observed_auc": auc, "null_auc": null_auc, "p_value": p_value}
        summary[method] = json_ready_method(auc, null_auc, p_value)

    y_out, pihd_scores, out_groups = predictions["pihd"]
    y_wave, wavelet_scores, wave_groups = predictions["wavelet"]
    if not (np.array_equal(y_out, y_wave) and np.array_equal(out_groups, wave_groups)):
        raise RuntimeError("PIHD and Wavelet LOSO predictions are not aligned.")
    ci_low, ci_high, bootstrap_delta = cluster_bootstrap_delta(
        y_out, pihd_scores, wavelet_scores, out_groups, np.random.default_rng(SEED + 20260809))
    delta_auc, swap_null, paired_p = subject_swap_test(
        y_out, pihd_scores, wavelet_scores, out_groups, np.random.default_rng(SEED + 123456))

    comparison = {
        "pihd_pooled_auc": float(observed["pihd"]), "wavelet_pooled_auc": float(observed["wavelet"]),
        "delta_auc": delta_auc, "ci95_low": ci_low, "ci95_high": ci_high,
        "paired_prediction_swap_p": paired_p,
    }
    print("\nFINAL PERMUTATION RESULTS")
    for method in METHODS:
        result = summary[method]
        print(f"{METHOD_LABELS[method]:<12} AUC={result['observed_pooled_auc']:.4f}  "
              f"p={result['permutation_p']:.6f}  {result['significance']}")
    print(f"\nPIHD vs Wavelet: Delta AUC={delta_auc:+.4f}; 95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]; p={paired_p:.6f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, {
        "seed": SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOTSTRAP,
        "n_method_compare_perm": N_COMPARE_PERM, "methods": METHODS, "observed_auc": observed,
        "permutation_results": permutation_results, "pihd_vs_wavelet": {
            **comparison, "bootstrap_delta": bootstrap_delta, "permutation_delta": swap_null},
        "summary": summary, "labels": y, "groups": groups, "subject_ids": data["subject_ids"],
    }, allow_pickle=True)
    json_payload = {"seed": SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOTSTRAP,
                    "n_method_compare_perm": N_COMPARE_PERM, "methods": summary,
                    "pihd_vs_wavelet": comparison}
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved:\n{output_npy}\n{output_json}")


if __name__ == "__main__":
    main()