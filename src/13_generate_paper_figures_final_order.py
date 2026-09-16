# -*- coding: utf-8 -*-
"""Generate the data-driven figures used by the latest PIHD paper.

Inputs (all must already exist)
--------------------------------
results/activation_task_rest_9methods_seed{seed}.npy
results/json/loso_9methods_results_seed{seed}.json
results/npy/loso_permutation_9methods_seed{seed}.npy
results/per_trial_params_seed{seed}.npy

Outputs
-------
figures/Fig2_main_results.png/.pdf
figures/Fig3_permutation_validation_refined.png/.pdf
figures/Fig4_hrf_parameters_refined.png/.pdf

No decomposition or classification is rerun. This script only reads saved
results and creates figures. Run from any working directory:

    python 13_generate_paper_figures_latest.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SEEDS = (42, 123, 999)
METHODS = ("raw", "bandpass", "glm", "ica", "pca", "wavelet", "ssa", "emd", "pihd")
LABELS = ("Raw HbO", "Bandpass", "GLM", "ICA", "PCA", "Wavelet", "SSA", "EMD", "PIHD")
INK = "#202124"
GRAY = "#8A8F98"
LIGHT_GRAY = "#D9DDE3"
PIHD_BLUE = "#0072B2"       # Okabe--Ito blue
WAVELET_ORANGE = "#D55E00"  # Okabe--Ito vermillion
SEED_COLORS = ("#3B82B8", "#35A786", "#C982AA")
SEED_FILLS = ("#BFD9EA", "#BCE2D4", "#E7C4D8")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11.0,
    "axes.titlesize": 11.0,
    "axes.titleweight": "normal",
    "axes.labelsize": 11.0,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 11.0,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_results():
    activation, classification, permutation, parameters = {}, {}, {}, {}
    for seed in SEEDS:
        activation_path = RESULTS_DIR / f"activation_task_rest_9methods_seed{seed}.npy"
        classification_path = RESULTS_DIR / "json" / f"loso_9methods_results_seed{seed}.json"
        permutation_path = RESULTS_DIR / "npy" / f"loso_permutation_9methods_seed{seed}.npy"
        parameter_path = RESULTS_DIR / f"per_trial_params_seed{seed}.npy"
        for path in (activation_path, classification_path, permutation_path, parameter_path):
            if not path.exists():
                raise FileNotFoundError(f"Required result not found: {path}")
        activation[seed] = np.load(activation_path, allow_pickle=True).item()
        classification[seed] = json.loads(classification_path.read_text(encoding="utf-8"))
        permutation[seed] = np.load(permutation_path, allow_pickle=True).item()
        parameters[seed] = np.load(parameter_path, allow_pickle=True).item()
    return activation, classification, permutation, parameters


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def save_figure(fig, stem):
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    except PermissionError:
        # Windows locks a PDF while it is open in some readers.  Preserve the
        # completed figure under the first available fallback name instead of
        # aborting the entire multi-figure script.
        for suffix in ["_revised", "_revised_2", "_revised_3"]:
            pdf = FIGURES_DIR / f"{stem}{suffix}.pdf"
            try:
                fig.savefig(pdf, bbox_inches="tight", facecolor="white")
                break
            except PermissionError:
                continue
        else:
            raise PermissionError(
                f"Close the open PDF copies for {stem} and run the script again."
            )
        print(f"Original PDF is open; used fallback: figures/{pdf.name}")
    plt.close(fig)
    print(f"Saved: figures/{png.name}")
    print(f"Saved: figures/{pdf.name}")


def p_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_main_results(activation, classification, permutation):
    # Panel (a): two adjacent bars occupy 0.68 units and consecutive method
    # groups are separated by an exact 0.30-unit gap.
    paired_bar_width = 0.34
    group_step = 2 * paired_bar_width + 0.30
    x = np.arange(len(METHODS), dtype=float) * group_step
    me_mean, me_sd, mi_mean, mi_sd = [], [], [], []
    trial_mean, trial_sd, loso_mean, loso_sd, conservative_p = [], [], [], [], []

    for method in METHODS:
        me = [float(activation[s]["dz_results"][method]["me_dz"]) for s in SEEDS]
        mi = [float(activation[s]["dz_results"][method]["mi_dz"]) for s in SEEDS]
        trial = [float(classification[s]["trial_cv"][method]["auc_mean"]) for s in SEEDS]
        loso = [float(permutation[s]["summary"][method]["observed_pooled_auc"]) for s in SEEDS]
        pvals = [float(permutation[s]["summary"][method]["permutation_p"]) for s in SEEDS]
        me_m, me_s = mean_sd(me)
        mi_m, mi_s = mean_sd(mi)
        tr_m, tr_s = mean_sd(trial)
        lo_m, lo_s = mean_sd(loso)
        me_mean.append(me_m); me_sd.append(me_s)
        mi_mean.append(mi_m); mi_sd.append(mi_s)
        trial_mean.append(tr_m); trial_sd.append(tr_s)
        loso_mean.append(lo_m); loso_sd.append(lo_s)
        conservative_p.append(max(pvals))

    # Baselines are deterministic; only PIHD has meaningful across-seed SD.
    me_err = np.zeros(len(METHODS)); me_err[-1] = me_sd[-1]
    mi_err = np.zeros(len(METHODS)); mi_err[-1] = mi_sd[-1]
    trial_err = np.zeros(len(METHODS)); trial_err[-1] = trial_sd[-1]
    loso_err = np.zeros(len(METHODS)); loso_err[-1] = loso_sd[-1]

    # Premium neutral/blue hierarchy.  ME and MI are separated by fill colour
    # rather than hatching, which keeps the panel clean at print resolution.
    me_colors = ["#D9D9D9"] * (len(METHODS) - 1) + ["#005B96"]
    mi_colors = ["#BDBDBD"] * (len(METHODS) - 1) + ["#4B9CD3"]
    single_colors = ["#C0C0C0"] * (len(METHODS) - 1) + ["#005B96"]
    edge = "#333333"
    reference = "#7F7F7F"

    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.titleweight": "normal",
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
    }):
        # Native ICASSP double-column width: no hidden down-scaling in LaTeX.
        fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65))
        width = paired_bar_width

        # (a) Task-minus-preceding-rest effects.
        ax = axes[0]
        ax.bar(x - width / 2, me_mean, width, yerr=me_err,
               color=me_colors, edgecolor=edge, linewidth=1.0,
               capsize=2.5, error_kw={"elinewidth": 1.2, "capthick": 1.2,
                                      "ecolor": "#000000"},
               label="ME $-$ Rest 1", zorder=3)
        ax.bar(x + width / 2, mi_mean, width, yerr=mi_err,
               color=mi_colors, edgecolor=edge,
               linewidth=1.0, capsize=2.5,
               error_kw={"elinewidth": 1.2, "capthick": 1.2,
                          "ecolor": "#000000"},
               label="MI $-$ Rest 2", zorder=3)
        ax.axhline(0, color=reference, linestyle=(0, (4, 2.5)),
                   linewidth=1.2, zorder=1)

        # Panel (a) intentionally omits numeric labels: paired ME/MI values are
        # too dense at publication width.  Significance remains immediately
        # above the corresponding error-bar cap.
        effect_star_pad = 0.055
        me_star_y = me_mean[-1] + me_err[-1] + effect_star_pad
        mi_star_y = mi_mean[-1] + mi_err[-1] + effect_star_pad
        ax.text(x[-1] - width / 2, me_star_y, "***", ha="center",
                va="bottom", color="#000000", fontsize=8.5, fontweight="bold")
        ax.text(x[-1] + width / 2, mi_star_y, "***", ha="center",
                va="bottom", color="#000000", fontsize=8.5, fontweight="bold")
        ax.set_title("Task-minus-rest effect", pad=7)
        ax.text(-0.17, 1.055, "a", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
        ax.set_ylabel("Cohen's $d_z$")
        effect_bottom = min(float(np.min(np.asarray(me_mean) - me_err)),
                            float(np.min(np.asarray(mi_mean) - mi_err))) - 0.10
        effect_annotation_top = max(me_star_y, mi_star_y)
        effect_top = effect_annotation_top + 0.08 * (effect_annotation_top - effect_bottom)
        ax.set_ylim(effect_bottom, effect_top)
        ax.legend(loc="upper left", frameon=False, handlelength=1.4,
                  labelspacing=0.25, borderaxespad=0.25)

        # (b) Trial-level AUC. Numerical labels are intentionally omitted;
        # exact values are reported in the paper/table.
        ax = axes[1]
        ax.bar(x, trial_mean, width=0.68, yerr=trial_err,
               color=single_colors, edgecolor=edge, linewidth=1.0,
               capsize=2.5, error_kw={"elinewidth": 1.2, "capthick": 1.2,
                                      "ecolor": "#000000"}, zorder=3)
        ax.axhline(0.5, color=reference, linestyle=(0, (4, 2.5)), linewidth=1.2,
                   zorder=1, clip_on=False)
        ax.set_title("Repeated 5-fold CV", pad=7)
        ax.text(-0.17, 1.055, "b", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
        ax.set_ylabel("AUC")
        auc_bottom = 0.48
        trial_peak = float(np.max(np.asarray(trial_mean) + trial_err))
        trial_top = trial_peak + 0.08 * (trial_peak - auc_bottom)
        ax.set_ylim(auc_bottom, trial_top)
        ax.set_yticks(np.arange(0.50, 0.6501, 0.025))

        # (c) Pooled LOSO AUC with permutation significance. Numerical labels
        # are omitted to preserve visual hierarchy at publication width.
        ax = axes[2]
        ax.bar(x, loso_mean, width=0.68, yerr=loso_err,
               color=single_colors, edgecolor=edge, linewidth=1.0,
               capsize=2.5, error_kw={"elinewidth": 1.2, "capthick": 1.2,
                                      "ecolor": "#000000"}, zorder=3)
        ax.axhline(0.5, color=reference, linestyle=(0, (4, 2.5)), linewidth=1.2,
                   zorder=1, clip_on=False)
        auc_star_pad = 0.004
        loso_star_y = []
        for xi, value, err, pval in zip(x, loso_mean, loso_err, conservative_p):
            stars = p_stars(pval)
            if stars:
                star_y = value + err + auc_star_pad
                loso_star_y.append(star_y)
                ax.text(xi, star_y, stars, ha="center",
                        va="bottom", color="#000000", fontsize=8.5,
                        fontweight="bold")
        ax.set_title("Pooled LOSO", pad=7)
        ax.text(-0.17, 1.055, "c", transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
        ax.set_ylabel("AUC")
        loso_annotation_top = max(loso_star_y)
        loso_top = loso_annotation_top + 0.08 * (loso_annotation_top - auc_bottom)
        ax.set_ylim(auc_bottom, loso_top)
        ax.set_yticks(np.arange(0.50, 0.6501, 0.025))

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(LABELS, rotation=38, ha="right",
                               rotation_mode="anchor")
            ax.grid(axis="y", color="#E0E0E0", linestyle=":", linewidth=1.0,
                    alpha=1.0, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#000000")
            ax.spines["bottom"].set_color("#000000")
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)
            ax.tick_params(width=1.2, length=4, direction="out")
        # Explain the reference line once outside all panels, avoiding overlap
        # with bars and preserving space inside the data region.
        chance_handle = Line2D([0], [0], color=reference,
                               linestyle=(0, (4, 2.5)), linewidth=1.2)
        fig.legend([chance_handle], ["Dashed line: chance level (AUC = 0.50)"],
                   loc="lower center", bbox_to_anchor=(0.5, -0.005),
                   frameon=False, handlelength=2.1, fontsize=7.0)
        fig.tight_layout(w_pad=1.55, pad=0.8, rect=(0, 0.075, 1, 1))
        save_figure(fig, "Fig2_main_results")


def plot_permutation_validation(permutation):
    # Single-column layout. Use an 8-point plotting scale so the labels remain
    # legible but do not dominate the 9-point ICASSP body text.
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.15))

    ax = axes[0]
    bins = np.linspace(0.40, 0.60, 32)
    for seed, color in zip(SEEDS, SEED_COLORS):
        null_auc = np.asarray(permutation[seed]["permutation_results"]["pihd"]["null_auc"], float)
        observed = float(permutation[seed]["summary"]["pihd"]["observed_pooled_auc"])
        ax.hist(null_auc[np.isfinite(null_auc)], bins=bins, density=True, histtype="step",
                linewidth=1.0, color=color, label=f"Seed {seed}")
        ax.axvline(observed, color=color, linestyle=(0, (3, 2)), linewidth=1.0)
    ax.set_xlabel("Permuted pooled LOSO AUC", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("a  PIHD label-permutation null", loc="left", pad=3, fontsize=8)
    density_top = ax.get_ylim()[1]
    ax.set_ylim(0, density_top / 0.80)
    ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.3,
              columnspacing=0.6, borderaxespad=0.35, fontsize=7.4)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)

    ax = axes[1]
    deltas, lower, upper, pvals = [], [], [], []
    for seed in SEEDS:
        comp = permutation[seed]["pihd_vs_wavelet"]
        deltas.append(float(comp["delta_auc"]))
        lower.append(float(comp["ci95_low"]))
        upper.append(float(comp["ci95_high"]))
        pvals.append(float(comp.get("paired_prediction_swap_p", comp.get("paired_permutation_p"))))
    deltas = np.asarray(deltas)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    positions = np.arange(1, len(SEEDS) + 1)
    yerr = np.vstack([deltas - lower, upper - deltas])
    ax.errorbar(positions, deltas, yerr=yerr, fmt="o", markersize=4.8,
                color=PIHD_BLUE, ecolor=INK, capsize=2.8, linewidth=0.9)
    ax.axhline(0, color=INK, linestyle=(0, (3, 2)), linewidth=0.7)
    for xpos, pval, hi in zip(positions, pvals, upper):
        ax.text(xpos, hi + 0.004, f"$p$={pval:.3f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"Seed {seed}" for seed in SEEDS])
    ax.set_ylabel(r"$\Delta$AUC (PIHD $-$ Wavelet)", fontsize=8)
    ax.set_title("b  Subject-cluster bootstrap 95% CI", loc="left", pad=3, fontsize=8)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(min(0, min(lower)) - 0.01, max(upper) + 0.025)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)

    fig.tight_layout(h_pad=0.45, pad=0.30)
    save_figure(fig, "Fig4_permutation_validation")


def plot_parameter_stability(parameters):
    tau = [np.asarray(parameters[s]["tau_list"], float) for s in SEEDS]
    eps = [np.asarray(parameters[s]["eps_list"], float) for s in SEEDS]
    for seed, tvals, evals in zip(SEEDS, tau, eps):
        if len(tvals) != 480 or len(evals) != 480:
            raise ValueError(f"Seed {seed}: expected 480 parameter values, got {len(tvals)} and {len(evals)}")

    # This file is inserted at one-column width. Generate it natively at that
    # width instead of shrinking a two-column canvas, which previously made
    # its labels about half the size of those in Figs. 2 and 4.
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.65))
    positions = np.arange(1, len(SEEDS) + 1)

    def violin_panel(ax, arrays, ylabel, title, physiological_range, ylim):
        parts = ax.violinplot(arrays, positions=positions, widths=0.72,
                              showmeans=False, showmedians=False, showextrema=False)
        for body, fill, edge in zip(parts["bodies"], SEED_FILLS, SEED_COLORS):
            body.set_facecolor(fill)
            body.set_edgecolor(edge)
            body.set_linewidth(0.9)
            body.set_alpha(0.92)
        for pos, values, edge in zip(positions, arrays, SEED_COLORS):
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            mean = values.mean()
            box_width = 0.14
            rect = plt.Rectangle((pos - box_width / 2, q1), box_width, q3 - q1,
                                 facecolor="white", edgecolor=edge, linewidth=0.8, zorder=3)
            ax.add_patch(rect)
            ax.hlines(median, pos - box_width / 2, pos + box_width / 2,
                      color=INK, linewidth=1.0, zorder=4)
            ax.scatter(pos, mean, color=INK, marker="D", s=11, zorder=5)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(seed) for seed in SEEDS], fontsize=7)
        ax.set_xlabel("Seed", fontsize=7.3)
        ax.set_ylabel(ylabel, fontsize=7.3)
        ax.set_title(title, loc="left", pad=3, fontsize=7.5)
        ax.tick_params(labelsize=7, width=0.6, length=2.5)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.4, alpha=0.50)
        ax.set_axisbelow(True)

    violin_panel(axes[0], tau, r"Venous transit time $\tau$ (s)",
                 r"a  Per-trial $\tau$ by seed", (2.0, 6.0), (2.0, 6.0))
    violin_panel(axes[1], eps, r"Extraction-related parameter $\varepsilon$",
                 r"b  Per-trial $\varepsilon$ by seed", (0.3, 1.5), (0.3, 1.5))
    axes[0].scatter([], [], color=INK, marker="D", s=11, label="Mean")
    axes[0].plot([], [], color=INK, linewidth=1.0, label="Median")
    axes[0].legend(frameon=False, loc="lower left", handlelength=1.0,
                   labelspacing=0.2, fontsize=6.8)
    fig.tight_layout(w_pad=0.55, pad=0.35)
    save_figure(fig, "Fig3_hrf_parameters")


def main():
    activation, classification, permutation, parameters = load_results()
    # Fig. 2 was manually curated and is intentionally not overwritten here.
    if not (FIGURES_DIR / "Fig2_main_results.pdf").exists():
        print("[WARN] Curated Fig2_main_results.pdf is absent; continuing.")
    plot_permutation_validation(permutation)
    plot_parameter_stability(parameters)
    print("Done. No model training or decomposition was rerun.")


if __name__ == "__main__":
    main()
