"""Read-only internal-consistency checks for the archived final results."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "result"
SEEDS = (42, 123, 999)
METHODS = ("raw", "bandpass", "glm", "ica", "pca", "wavelet", "ssa", "emd", "pihd")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_npy(path):
    return np.load(path, allow_pickle=True).item()


def close(a, b, label, atol=1e-12):
    if not np.allclose(a, b, atol=atol, rtol=1e-10, equal_nan=True):
        raise AssertionError(f"{label}: {a!r} != {b!r}")


def main():
    loso_auc = []
    me_dz = []
    mi_dz = []
    for seed in SEEDS:
        cls = load_npy(ARCHIVE / "07_loso_classification" / "npy" /
                       f"loso_classification_9methods_seed{seed}.npy")
        perm = load_npy(ARCHIVE / "08.5_loso_permutation" / "npy" /
                        f"loso_permutation_9methods_seed{seed}.npy")
        perm_json = load_json(ARCHIVE / "08.5_loso_permutation" / "json" /
                              f"loso_permutation_9methods_seed{seed}.json")
        activation = load_npy(ARCHIVE / "08.6_task_rest" /
                              f"activation_task_rest_9methods_seed{seed}.npy")
        activation_json = load_json(ARCHIVE / "08.6_task_rest" / "json" /
                                    f"activation_task_rest_9methods_seed{seed}.json")
        ablation = load_npy(ARCHIVE / "06.5_key_ablation" / "npy" /
                            f"ablation_key3_48subj_seed{seed}.npy")

        for obj, name in ((cls, "classification"), (activation, "activation"),
                          (ablation, "ablation")):
            assert int(obj["n_subjects"]) == 48, (name, seed, "n_subjects")
            assert int(obj["n_trials"]) == 480, (name, seed, "n_trials")
        assert tuple(tuple(activation["windows_seconds"][k]) for k in
                     ("rest1", "me", "rest2", "mi")) == ((2, 7), (8, 16), (17, 23), (24, 32))
        assert activation["diff_method"] == "task minus preceding rest"
        assert perm["n_permutations"] == 1000
        assert len(np.unique(cls["groups"])) == 48

        for method in METHODS:
            close(perm["observed_auc"][method],
                  perm_json["methods"][method]["observed_pooled_auc"],
                  f"permutation JSON/NPY {seed} {method}")
            close(activation["dz_results"][method]["me_dz"],
                  activation_json[method]["me_dz"],
                  f"activation ME JSON/NPY {seed} {method}", atol=1e-7)
            close(activation["dz_results"][method]["mi_dz"],
                  activation_json[method]["mi_dz"],
                  f"activation MI JSON/NPY {seed} {method}", atol=1e-7)

        full = ablation["results"]["Full PIHD"]
        recomputed_auc = roc_auc_score(full["loso_true"], full["loso_prob"])
        close(recomputed_auc, full["auc_loso"], f"recomputed pooled AUC {seed}")
        for task in ("me", "mi"):
            values = np.asarray(activation["dz_results"]["pihd"][f"{task}_subject_values"])
            recomputed_dz = values.mean() / values.std(ddof=1)
            close(recomputed_dz, activation["dz_results"]["pihd"][f"{task}_dz"],
                  f"recomputed participant-level {task.upper()} dz {seed}", atol=1e-8)
        close(full["auc_loso"], perm["observed_auc"]["pihd"],
              f"full PIHD/LOSO AUC {seed}")
        close(full["me_d"], activation["dz_results"]["pihd"]["me_dz"],
              f"full PIHD/activation ME {seed}", atol=1e-6)
        close(full["mi_d"], activation["dz_results"]["pihd"]["mi_dz"],
              f"full PIHD/activation MI {seed}", atol=1e-6)
        assert perm_json["pihd_vs_wavelet"]["ci95_low"] < 0 < perm_json["pihd_vs_wavelet"]["ci95_high"]
        assert perm_json["pihd_vs_wavelet"]["paired_prediction_swap_p"] >= 0.05
        loso_auc.append(full["auc_loso"])
        me_dz.append(full["me_d"])
        mi_dz.append(full["mi_d"])

    smooth = load_npy(ARCHIVE / "06.7_smoothness_statistics" / "npy" /
                      "smoothness_control_v2_summary.npy")
    close(smooth["full_pihd_auc_by_seed"], loso_auc, "smoothness PIHD references")
    assert smooth["participant_cluster_bootstrap_ci95"][0] > 0
    assert smooth["participant_paired_swap_p_two_sided"] < 0.05

    def mean_sd(values):
        return np.mean(values), np.std(values, ddof=1)
    auc_mean, auc_sd = mean_sd(loso_auc)
    me_mean, me_sd = mean_sd(me_dz)
    mi_mean, mi_sd = mean_sd(mi_dz)
    print("PASS: archived final results are internally consistent.")
    print(f"PIHD pooled LOSO AUC: {auc_mean:.3f} +/- {auc_sd:.3f}")
    print(f"Task-minus-rest: ME dz={me_mean:.2f} +/- {me_sd:.2f}; "
          f"MI dz={mi_mean:.2f} +/- {mi_sd:.2f}")
    print("Scope: cross-file checks only; no experiment was rerun.")


if __name__ == "__main__":
    main()
