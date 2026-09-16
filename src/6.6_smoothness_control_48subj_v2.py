# -*- coding: utf-8 -*-
"""Gradient-strength-matched smoothness control for PIHD (48 participants), v2.

Scientific question
-------------------
Does the full PIHD result require the Balloon--Windkessel ODE, or can a
generic smoothness penalty produce a similar result?

This script replaces L_ode with

    L_smooth = mean((d^2 h(t) / dt^2)^2)

and keeps the data, networks, Stage-1 procedure, remaining losses, training
epochs, features, classifier, and LOSO protocol identical to the key
48-participant ablation.  It does NOT rerun Full PIHD.  Full-PIHD held-out
predictions are read from the existing Step 6.5 result.

To avoid choosing a favorable arbitrary coefficient, lambda_smooth is fixed
per trial by matching the total gradient norm of L_smooth to that of L_ode
over FVQNet, FinNet, tau, and epsilon parameters.  Matching occurs at the
first numerically valid epoch from epoch 20 onward.  There is no coefficient
clipping.  L_ode is evaluated once only to obtain this scalar magnitude and
is never included in the control's optimization objective.

Outputs
-------
results/npy/smoothness_control_v2_48subj_seed{SEED}.npy
results/json/smoothness_control_v2_48subj_seed{SEED}.json
results/npy/smoothness_control_v2_48subj_seed{SEED}_checkpoint.npy

Usage (three terminals are fine)
--------------------------------
python 6.6_smoothness_control_48subj_v2.py 42
python 6.6_smoothness_control_48subj_v2.py 123
python 6.6_smoothness_control_48subj_v2.py 999
"""

import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.special import gamma as gamma_fn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA, E0 = 0.32, 0.34
SAFE = 1e-6
N_EPOCHS_S1 = 400
N_EPOCHS_S2 = 800
SMOOTH_MATCH_EPOCH = 20
SMOOTH_MATCH_LAST_EPOCH = 100
SMOOTH_GRAD_EPS = 1e-20
LR = 2e-3
N_SUBJECTS = 48
N_PTS = 80
NORM_FLOOR = 0.05
N_CV_REPEATS = 20
N_LABEL_PERM = 1000
N_PAIRED_PERM = 10000
N_CLUSTER_BOOT = 5000
CHECKPOINT_EVERY = 20
SYS_FREQS = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 1.00]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FEAT_PATH = os.path.join(REPO_ROOT, "data", "eeg_features_all_v6.npy")
NPY_DIR = os.path.join(REPO_ROOT, "results", "npy")
JSON_DIR = os.path.join(REPO_ROOT, "results", "json")
os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

FULL_RESULT_PATH = os.path.join(
    NPY_DIR, f"ablation_key3_48subj_seed{SEED}.npy"
)
CHECKPOINT_PATH = os.path.join(
    NPY_DIR, f"smoothness_control_v2_48subj_seed{SEED}_checkpoint.npy"
)
OUTPUT_PATH = os.path.join(
    NPY_DIR, f"smoothness_control_v2_48subj_seed{SEED}.npy"
)
JSON_PATH = os.path.join(
    JSON_DIR, f"smoothness_control_v2_48subj_seed{SEED}.json"
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")


# -----------------------------------------------------------------------------
# PIHD modules (identical to Step 6.5)
# -----------------------------------------------------------------------------
class FVQNet(nn.Module):
    def __init__(self, hidden=(20, 40, 40, 20)):
        super().__init__()
        layers = []
        prev = 1
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            prev = width
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        x = t
        for index, layer in enumerate(self.net):
            x = layer(x)
            if index < len(self.net) - 1:
                x = nn.functional.softplus(x)
        f = nn.functional.softplus(x[:, 0:1])
        v = nn.functional.softplus(x[:, 1:2])
        q = 1.0 + x[:, 2:3]
        return f, v, q


class FinNet(nn.Module):
    def __init__(self, hidden=(16, 16)):
        super().__init__()
        layers = []
        prev = 1
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.Softplus()])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        return nn.functional.softplus(self.net(t))


class SystemicNoise(nn.Module):
    def __init__(self, freqs, device):
        super().__init__()
        count = len(freqs)
        self.cos_coeffs = nn.Parameter(torch.zeros(count, device=device))
        self.sin_coeffs = nn.Parameter(torch.zeros(count, device=device))
        self.bias = nn.Parameter(torch.tensor(0.0, device=device))
        self.drift = nn.Parameter(torch.tensor(0.0, device=device))
        self.register_buffer(
            "freq_t", torch.tensor(freqs, dtype=torch.float32, device=device)
        )

    def forward(self, t, cos_c=None, sin_c=None, b=None, d=None):
        cc = self.cos_coeffs if cos_c is None else cos_c
        sc = self.sin_coeffs if sin_c is None else sin_c
        bb = self.bias if b is None else b
        dd = self.drift if d is None else d
        angles = 2 * np.pi * t * self.freq_t.unsqueeze(0)
        return (
            torch.cos(angles) @ cc.unsqueeze(1)
            + torch.sin(angles) @ sc.unsqueeze(1)
            + bb
            + dd * t
        )


def make_hrf_prior_torch(t_np, onset=8.0, duration=8.0):
    mask = (t_np >= onset) & (t_np < onset + duration)
    dt = t_np[1] - t_np[0]
    box = np.where(mask, 1.0, 0.0)
    t_kernel = np.arange(0, 8, dt)
    kernel = (
        (t_kernel ** (2.5 - 1))
        * (1.5**2.5)
        * np.exp(-1.5 * t_kernel)
        / gamma_fn(2.5)
    )
    kernel = kernel / (kernel.sum() + 1e-10)
    convolved = np.convolve(box, kernel, mode="same")
    return convolved / (convolved.max() + 1e-10) * 0.5


def bw_residual(t, f_in, f, v, q, tau, eps):
    """Used only once to match regularization gradient strength."""
    ones = torch.ones_like(t)
    df = torch.autograd.grad(f, t, ones, create_graph=True)[0]
    dv = torch.autograd.grad(v, t, ones, create_graph=True)[0]
    dq = torch.autograd.grad(q, t, ones, create_graph=True)[0]
    f_safe = torch.clamp(f, min=SAFE)
    v_safe = torch.clamp(v, min=SAFE)
    eps_safe = torch.clamp(eps, min=SAFE)
    extraction_arg = torch.clamp(
        1.0 - (1.0 / eps_safe) * (1.0 - 1.0 / f_safe), min=-0.99
    )
    extraction = E0 / (1.0 + E0 * extraction_arg)
    r_f = df - (f_in - (f - 1) / tau)
    r_v = dv - (f - v_safe ** (1 / ALPHA)) / tau
    r_q = dq - (f * extraction - v_safe ** (1 / ALPHA) * q / v_safe) / tau
    return r_f, r_v, r_q


def second_derivative_loss(t, h):
    ones = torch.ones_like(h)
    dh = torch.autograd.grad(h, t, ones, create_graph=True)[0]
    d2h = torch.autograd.grad(dh, t, torch.ones_like(dh), create_graph=True)[0]
    return torch.mean(d2h**2)


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    squared = torch.zeros((), device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + torch.sum(gradient.detach() ** 2)
    return torch.sqrt(squared + 1e-24)


def smoothness_control(t_np, hbo_norm, device=DEVICE):
    """Run the matched non-physical control for one trial."""
    set_seed(SEED)
    t_tensor = torch.tensor(
        t_np, dtype=torch.float32, device=device
    ).unsqueeze(1)
    h_tensor = torch.tensor(
        hbo_norm, dtype=torch.float32, device=device
    ).unsqueeze(1)

    rest_mask_np = (t_np < 7.5) | ((t_np > 16.5) & (t_np < 23.5))
    task_mask_np = ((t_np >= 8) & (t_np < 16)) | ((t_np >= 24) & (t_np < 32))
    rest_index = torch.tensor(rest_mask_np, dtype=torch.bool, device=device)
    task_index = torch.tensor(task_mask_np, dtype=torch.bool, device=device)

    # Stage 1: identical rest-only Fourier fit.
    noise_stage1 = SystemicNoise(SYS_FREQS, device).to(device)
    optimizer1 = torch.optim.Adam(noise_stage1.parameters(), lr=LR)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=N_EPOCHS_S1, eta_min=1e-5
    )
    best_stage1_loss = float("inf")
    best_stage1_state = None
    for _ in range(N_EPOCHS_S1):
        optimizer1.zero_grad()
        noise_prediction = noise_stage1(t_tensor)
        loss = torch.mean(
            (noise_prediction[rest_index] - h_tensor[rest_index]) ** 2
        )
        loss = (
            loss
            + 0.001 * torch.mean(noise_prediction**2)
            + 0.01 * noise_stage1.drift**2
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        optimizer1.step()
        scheduler1.step()
        if loss.item() < best_stage1_loss:
            best_stage1_loss = loss.item()
            best_stage1_state = {
                key: value.detach().clone()
                for key, value in noise_stage1.state_dict().items()
            }
    if best_stage1_state is None:
        raise RuntimeError("Stage 1 did not produce a finite state")
    noise_stage1.load_state_dict(best_stage1_state)
    fixed_cos = best_stage1_state["cos_coeffs"]
    fixed_sin = best_stage1_state["sin_coeffs"]

    # Stage 2: replace L_ode by a generic curvature penalty.
    t_tensor.requires_grad_(True)
    fvq = FVQNet().to(device)
    fin = FinNet().to(device)
    noise_bias = nn.Parameter(best_stage1_state["bias"].clone())
    noise_drift = nn.Parameter(best_stage1_state["drift"].clone())
    tau_raw = nn.Parameter(torch.tensor(0.0, device=device))
    eps_raw = nn.Parameter(torch.tensor(0.0, device=device))
    parameters = (
        list(fvq.parameters())
        + list(fin.parameters())
        + [tau_raw, eps_raw, noise_bias, noise_drift]
    )
    optimizer2 = torch.optim.Adam(parameters, lr=LR)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=N_EPOCHS_S2, eta_min=1e-5
    )
    hrf_tensor = torch.tensor(
        make_hrf_prior_torch(t_np), dtype=torch.float32, device=device
    ).unsqueeze(1)

    lambda_smooth = 1.0
    lambda_was_matched = False
    match_epoch = -1
    ode_gradient_norm = np.nan
    smooth_gradient_norm = np.nan
    matched_gradient_relative_error = np.nan
    best_stage2_loss = float("inf")
    best_stage2_state = None

    for epoch in range(N_EPOCHS_S2):
        optimizer2.zero_grad()
        f_in = fin(t_tensor)
        f, v, q = fvq(t_tensor)
        h_hemodynamic = v - q
        noise_prediction = noise_stage1(
            t_tensor,
            cos_c=fixed_cos,
            sin_c=fixed_sin,
            b=noise_bias,
            d=noise_drift,
        )
        reconstructed = h_hemodynamic + noise_prediction

        loss_data = torch.mean((reconstructed - h_tensor) ** 2)
        loss_smooth_raw = second_derivative_loss(t_tensor, h_hemodynamic)

        t_zero = torch.tensor([[0.0]], device=device, requires_grad=True)
        f_zero, v_zero, q_zero = fvq(t_zero)
        f_in_zero = fin(t_zero)
        loss_initial = torch.mean(
            (f_zero - 1) ** 2
            + (v_zero - 1) ** 2
            + (q_zero - 1) ** 2
            + f_in_zero**2
        )
        loss_rest = 2.0 * torch.mean(f_in[rest_index.unsqueeze(1)] ** 2)
        loss_hrf = (
            torch.mean((f_in - hrf_tensor) ** 2)
            if epoch < 400
            else torch.zeros((), device=device)
        )
        loss_variance = 0.5 * torch.exp(
            -5.0 * torch.var(h_hemodynamic[task_index.unsqueeze(1)])
        )

        # Match once at the first valid epoch in [20, 100].  No clipping is
        # applied: lambda * ||grad L_smooth|| equals ||grad L_ode|| up to
        # floating-point precision.  The ODE gradient is used only as a scalar
        # magnitude reference and never enters total_loss.
        if (
            not lambda_was_matched
            and SMOOTH_MATCH_EPOCH <= epoch <= SMOOTH_MATCH_LAST_EPOCH
        ):
            tau = 0.5 + 5.5 * torch.sigmoid(tau_raw)
            eps = 0.1 + 1.9 * torch.sigmoid(eps_raw)
            r_f, r_v, r_q = bw_residual(t_tensor, f_in, f, v, q, tau, eps)
            loss_ode_reference = (
                torch.mean(r_f**2) + torch.mean(r_v**2) + torch.mean(r_q**2)
            )
            physics_parameters = (
                list(fvq.parameters())
                + list(fin.parameters())
                + [tau_raw, eps_raw]
            )
            ode_norm = gradient_norm(loss_ode_reference, physics_parameters)
            smooth_norm = gradient_norm(loss_smooth_raw, physics_parameters)
            raw_ratio = ode_norm / smooth_norm
            if (
                torch.isfinite(ode_norm)
                and torch.isfinite(smooth_norm)
                and torch.isfinite(raw_ratio)
                and smooth_norm > SMOOTH_GRAD_EPS
                and raw_ratio > 0
            ):
                lambda_smooth = float(raw_ratio.item())
                lambda_was_matched = True
                match_epoch = epoch
                ode_gradient_norm = float(ode_norm.item())
                smooth_gradient_norm = float(smooth_norm.item())
                matched_norm = lambda_smooth * smooth_gradient_norm
                matched_gradient_relative_error = abs(
                    matched_norm - ode_gradient_norm
                ) / max(ode_gradient_norm, 1e-30)
                # Losses before and after coefficient matching are not
                # comparable, so model selection restarts here.
                best_stage2_loss = float("inf")
                best_stage2_state = None

        total_loss = (
            loss_data
            + lambda_smooth * loss_smooth_raw
            + 0.5 * loss_initial
            + loss_rest
            + loss_hrf
            + loss_variance
        )
        if not torch.isfinite(total_loss):
            continue
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer2.step()
        scheduler2.step()

        if total_loss.item() < best_stage2_loss:
            best_stage2_loss = total_loss.item()
            best_stage2_state = {
                key: value.detach().clone()
                for key, value in fvq.state_dict().items()
            }

    if not lambda_was_matched:
        raise RuntimeError(
            "Could not obtain a finite, non-zero smoothness gradient for "
            f"exact matching between epochs {SMOOTH_MATCH_EPOCH} and "
            f"{SMOOTH_MATCH_LAST_EPOCH}."
        )
    if best_stage2_state is None:
        raise RuntimeError("Stage 2 did not produce a finite state")
    fvq.load_state_dict(best_stage2_state)
    fvq.eval()
    with torch.no_grad():
        t_tensor.requires_grad_(False)
        _, v, q = fvq(t_tensor)
        signal = (v - q).cpu().numpy().ravel()
    signal = signal - np.mean(signal[rest_mask_np])
    diagnostics = {
        "lambda_smooth": lambda_smooth,
        "matched": lambda_was_matched,
        "match_epoch": match_epoch,
        "ode_gradient_norm": ode_gradient_norm,
        "smooth_gradient_norm_raw": smooth_gradient_norm,
        "matched_gradient_relative_error": matched_gradient_relative_error,
    }
    return signal, diagnostics


# -----------------------------------------------------------------------------
# Features, inference, and statistics
# -----------------------------------------------------------------------------
def extract_features(signal, t):
    me_mask = (t >= 8) & (t < 16)
    mi_mask = (t >= 24) & (t < 32)
    me_signal, mi_signal = signal[me_mask], signal[mi_mask]
    me_t, mi_t = t[me_mask], t[mi_mask]
    return np.asarray(
        [
            np.mean(me_signal),
            np.mean(mi_signal),
            np.max(me_signal),
            np.max(mi_signal),
            np.polyfit(me_t - np.mean(me_t), me_signal, 1)[0],
            np.polyfit(mi_t - np.mean(mi_t), mi_signal, 1)[0],
            np.var(me_signal),
            np.var(mi_signal),
        ],
        dtype=float,
    )


def pooled_loso_predictions(features, labels, groups):
    logo = LeaveOneGroupOut()
    probabilities = np.full(len(labels), np.nan, dtype=float)
    predictions = np.full(len(labels), -1, dtype=int)
    fold_auc = []
    for train_index, test_index in logo.split(features, labels, groups):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_index])
        x_test = scaler.transform(features[test_index])
        classifier = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        classifier.fit(x_train, labels[train_index])
        probabilities[test_index] = classifier.predict_proba(x_test)[:, 1]
        predictions[test_index] = classifier.predict(x_test)
        if len(np.unique(labels[test_index])) > 1:
            fold_auc.append(
                roc_auc_score(labels[test_index], probabilities[test_index])
            )
        else:
            fold_auc.append(np.nan)
    if np.isnan(probabilities).any() or (predictions < 0).any():
        raise RuntimeError("Missing pooled LOSO predictions")
    return probabilities, predictions, np.asarray(fold_auc, dtype=float)


def trial_cv_metrics(features, labels):
    accuracy_values, auc_values, f1_values = [], [], []
    for repeat in range(N_CV_REPEATS):
        splitter = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=repeat
        )
        for train_index, test_index in splitter.split(features, labels):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(features[train_index])
            x_test = scaler.transform(features[test_index])
            classifier = LogisticRegression(
                C=1.0, max_iter=1000, random_state=repeat
            )
            classifier.fit(x_train, labels[train_index])
            probability = classifier.predict_proba(x_test)[:, 1]
            prediction = classifier.predict(x_test)
            accuracy_values.append(
                accuracy_score(labels[test_index], prediction)
            )
            auc_values.append(roc_auc_score(labels[test_index], probability))
            f1_values.append(f1_score(labels[test_index], prediction))
    return (
        float(np.mean(accuracy_values)),
        float(np.mean(auc_values)),
        float(np.mean(f1_values)),
    )


def label_permutation_test(features, labels, groups, observed_auc):
    rng = np.random.RandomState(SEED + 6601)
    null_auc = np.empty(N_LABEL_PERM, dtype=float)
    for index in range(N_LABEL_PERM):
        permuted_labels = rng.permutation(labels)
        probability, _, _ = pooled_loso_predictions(
            features, permuted_labels, groups
        )
        null_auc[index] = roc_auc_score(permuted_labels, probability)
        if (index + 1) % 100 == 0:
            print(f"  Label permutations: {index + 1}/{N_LABEL_PERM}", flush=True)
    p_value = (1 + np.sum(null_auc >= observed_auc)) / (N_LABEL_PERM + 1)
    return null_auc, float(p_value)


def paired_method_comparison(labels, groups, full_probability, control_probability):
    unique_groups = np.asarray(sorted(np.unique(groups)))
    observed_delta = float(
        roc_auc_score(labels, full_probability)
        - roc_auc_score(labels, control_probability)
    )

    rng_perm = np.random.RandomState(SEED + 6602)
    null_delta = np.empty(N_PAIRED_PERM, dtype=float)
    for iteration in range(N_PAIRED_PERM):
        probability_a = full_probability.copy()
        probability_b = control_probability.copy()
        swap_groups = unique_groups[rng_perm.rand(len(unique_groups)) < 0.5]
        for group in swap_groups:
            mask = groups == group
            temporary = probability_a[mask].copy()
            probability_a[mask] = probability_b[mask]
            probability_b[mask] = temporary
        null_delta[iteration] = (
            roc_auc_score(labels, probability_a)
            - roc_auc_score(labels, probability_b)
        )
    p_two_sided = (
        1 + np.sum(np.abs(null_delta) >= abs(observed_delta))
    ) / (N_PAIRED_PERM + 1)

    rng_boot = np.random.RandomState(SEED + 6603)
    bootstrap_delta = np.empty(N_CLUSTER_BOOT, dtype=float)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for iteration in range(N_CLUSTER_BOOT):
        sampled_groups = rng_boot.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        sampled_index = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        sampled_labels = labels[sampled_index]
        if len(np.unique(sampled_labels)) < 2:
            bootstrap_delta[iteration] = np.nan
        else:
            bootstrap_delta[iteration] = (
                roc_auc_score(sampled_labels, full_probability[sampled_index])
                - roc_auc_score(sampled_labels, control_probability[sampled_index])
            )
    valid_bootstrap = bootstrap_delta[np.isfinite(bootstrap_delta)]
    ci_low, ci_high = np.percentile(valid_bootstrap, [2.5, 97.5])
    return {
        "observed_delta_auc": observed_delta,
        "paired_swap_p_two_sided": float(p_two_sided),
        "cluster_bootstrap_ci95": np.asarray([ci_low, ci_high], dtype=float),
        "paired_swap_null_delta": null_delta,
        "cluster_bootstrap_delta": bootstrap_delta,
    }


def save_checkpoint(records, subject_ids):
    np.save(
        CHECKPOINT_PATH,
        {
            "seed": SEED,
            "n_subjects": len(subject_ids),
            "n_completed": len(records["features"]),
            "records": records,
            "control": "second-derivative smoothness replacing L_ode",
        },
        allow_pickle=True,
    )


# -----------------------------------------------------------------------------
# Run decomposition control
# -----------------------------------------------------------------------------
if not os.path.exists(FEAT_PATH):
    raise FileNotFoundError(f"Feature file not found: {FEAT_PATH}")
if not os.path.exists(FULL_RESULT_PATH):
    raise FileNotFoundError(
        f"Full PIHD reference not found: {FULL_RESULT_PATH}\n"
        f"Run Step 6.5 for seed {SEED} first."
    )

feature_payload = np.load(FEAT_PATH, allow_pickle=True).item()
trials = feature_payload["trials"]
subjects = sorted({trial["subject"] for trial in trials})
if len(subjects) > N_SUBJECTS:
    selected_rng = np.random.RandomState(42)
    selected_subjects = list(
        selected_rng.choice(subjects, N_SUBJECTS, replace=False)
    )
else:
    selected_subjects = subjects
selected_trials = [
    trial for trial in trials if trial["subject"] in selected_subjects
]
print(f"Participants: {len(selected_subjects)}; trials: {len(selected_trials)}")

records = {
    "features": [],
    "labels": [],
    "groups": [],
    "me_trial_contrast": [],
    "mi_trial_contrast": [],
    "lambda_smooth": [],
    "lambda_matched": [],
    "match_epoch": [],
    "ode_gradient_norm": [],
    "smooth_gradient_norm_raw": [],
    "matched_gradient_relative_error": [],
}
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = np.load(CHECKPOINT_PATH, allow_pickle=True).item()
    if checkpoint.get("seed") != SEED:
        raise ValueError("Checkpoint seed mismatch")
    records = checkpoint["records"]
    print(f"Resuming after {len(records['features'])} completed trials")

t_grid = np.linspace(0, 32, N_PTS)
start_index = len(records["features"])
run_start = time.time()

for trial_index in range(start_index, len(selected_trials)):
    trial_start = time.time()
    trial = selected_trials[trial_index]
    hbo = np.interp(t_grid, trial["t"], trial["hbo_n"])
    baseline_mask = t_grid < 8
    baseline_mean = np.mean(hbo[baseline_mask])
    baseline_sd = max(np.std(hbo[baseline_mask]), NORM_FLOOR)
    hbo_normalized = (hbo - baseline_mean) / baseline_sd

    signal, match_diagnostics = smoothness_control(
        t_grid, hbo_normalized, device=DEVICE
    )
    lambda_smooth = match_diagnostics["lambda_smooth"]
    was_matched = match_diagnostics["matched"]
    features = extract_features(signal, t_grid)
    rest1 = (t_grid >= 2) & (t_grid < 7)
    me = (t_grid >= 8) & (t_grid < 16)
    rest2 = (t_grid >= 17) & (t_grid < 23)
    mi = (t_grid >= 24) & (t_grid < 32)

    records["features"].append(features)
    records["labels"].append(1 if trial["hand"] == "right" else 0)
    records["groups"].append(trial["subject"])
    records["me_trial_contrast"].append(
        float(np.mean(signal[me]) - np.mean(signal[rest1]))
    )
    records["mi_trial_contrast"].append(
        float(np.mean(signal[mi]) - np.mean(signal[rest2]))
    )
    records["lambda_smooth"].append(float(lambda_smooth))
    records["lambda_matched"].append(bool(was_matched))
    records["match_epoch"].append(int(match_diagnostics["match_epoch"]))
    records["ode_gradient_norm"].append(
        float(match_diagnostics["ode_gradient_norm"])
    )
    records["smooth_gradient_norm_raw"].append(
        float(match_diagnostics["smooth_gradient_norm_raw"])
    )
    records["matched_gradient_relative_error"].append(
        float(match_diagnostics["matched_gradient_relative_error"])
    )

    if (trial_index + 1) % CHECKPOINT_EVERY == 0 or trial_index == 0:
        save_checkpoint(records, selected_subjects)
        print(
            f"[{trial_index + 1:3d}/{len(selected_trials)}] "
            f"ME={records['me_trial_contrast'][-1]:+.3f} "
            f"MI={records['mi_trial_contrast'][-1]:+.3f} "
            f"lambda={lambda_smooth:.3g} "
            f"matched={was_matched} "
            f"match_epoch={match_diagnostics['match_epoch']} "
            f"relerr={match_diagnostics['matched_gradient_relative_error']:.2e} "
            f"({time.time() - trial_start:.1f}s)",
            flush=True,
        )

save_checkpoint(records, selected_subjects)

features = np.asarray(records["features"], dtype=float)
labels = np.asarray(records["labels"], dtype=int)
groups = np.asarray(records["groups"])
me_trial = np.asarray(records["me_trial_contrast"], dtype=float)
mi_trial = np.asarray(records["mi_trial_contrast"], dtype=float)
lambda_values = np.asarray(records["lambda_smooth"], dtype=float)
lambda_matched = np.asarray(records["lambda_matched"], dtype=bool)
match_epochs = np.asarray(records["match_epoch"], dtype=int)
ode_gradient_norms = np.asarray(records["ode_gradient_norm"], dtype=float)
smooth_gradient_norms = np.asarray(
    records["smooth_gradient_norm_raw"], dtype=float
)
match_relative_errors = np.asarray(
    records["matched_gradient_relative_error"], dtype=float
)

if len(features) != len(selected_trials):
    raise RuntimeError("Checkpoint/output does not contain every selected trial")
if not np.all(lambda_matched):
    raise RuntimeError("At least one trial lacks exact gradient-strength matching")
if not np.all(np.isfinite(lambda_values)) or np.any(lambda_values <= 0):
    raise RuntimeError("Invalid lambda_smooth value detected")
if np.max(match_relative_errors) > 1e-5:
    raise RuntimeError(
        "Gradient-strength matching relative error exceeded 1e-5"
    )

unique_subjects = np.asarray(sorted(np.unique(groups)))
me_subject = np.asarray(
    [np.mean(me_trial[groups == subject]) for subject in unique_subjects]
)
mi_subject = np.asarray(
    [np.mean(mi_trial[groups == subject]) for subject in unique_subjects]
)
me_dz = float(np.mean(me_subject) / (np.std(me_subject, ddof=1) + 1e-10))
mi_dz = float(np.mean(mi_subject) / (np.std(mi_subject, ddof=1) + 1e-10))
me_t, me_p = stats.ttest_1samp(me_subject, 0)
mi_t, mi_p = stats.ttest_1samp(mi_subject, 0)

cv_accuracy, cv_auc, cv_f1 = trial_cv_metrics(features, labels)
loso_probability, loso_prediction, loso_fold_auc = pooled_loso_predictions(
    features, labels, groups
)
loso_auc = float(roc_auc_score(labels, loso_probability))
loso_accuracy = float(accuracy_score(labels, loso_prediction))

print("\nRunning 1000 label permutations (decomposition is NOT rerun)...")
null_auc, permutation_p = label_permutation_test(
    features, labels, groups, loso_auc
)

full_payload = np.load(FULL_RESULT_PATH, allow_pickle=True).item()
full_result = full_payload["results"]["Full PIHD"]
full_labels = np.asarray(full_result["loso_true"], dtype=int)
full_groups = np.asarray(full_result["groups"])
full_probability = np.asarray(full_result["loso_prob"], dtype=float)
if not np.array_equal(labels, full_labels) or not np.array_equal(groups, full_groups):
    raise ValueError("Full PIHD and smoothness control trial order does not match")

comparison = paired_method_comparison(
    labels, groups, full_probability, loso_probability
)

result = {
    "seed": SEED,
    "n_subjects": len(unique_subjects),
    "n_trials": len(labels),
    "control_name": "Gradient-strength-matched second-derivative smoothness control v2",
    "control_definition": "mean((d2 h / dt2)^2), replacing L_ode",
    "weight_matching": {
        "method": (
            "Exact total gradient-norm match to L_ode over FVQNet, FinNet, "
            "tau, and epsilon; first valid Stage-2 epoch in [20, 100]"
        ),
        "calibration_only": True,
        "coefficient_clipping": False,
        "lambda_values": lambda_values,
        "matched_flags": lambda_matched,
        "match_epochs": match_epochs,
        "ode_gradient_norms": ode_gradient_norms,
        "smooth_gradient_norms_raw": smooth_gradient_norms,
        "matched_gradient_relative_errors": match_relative_errors,
    },
    "activation": {
        "me_dz": me_dz,
        "mi_dz": mi_dz,
        "me_t": float(me_t),
        "mi_t": float(mi_t),
        "me_p_raw": float(me_p),
        "mi_p_raw": float(mi_p),
        "me_subject_values": me_subject,
        "mi_subject_values": mi_subject,
    },
    "classification": {
        "trial_cv_accuracy": cv_accuracy,
        "trial_cv_auc": cv_auc,
        "trial_cv_f1": cv_f1,
        "pooled_loso_accuracy": loso_accuracy,
        "pooled_loso_auc": loso_auc,
        "loso_true": labels,
        "loso_prob": loso_probability,
        "loso_pred": loso_prediction,
        "loso_fold_auc": loso_fold_auc,
        "label_permutation_p": permutation_p,
        "label_permutation_null_auc": null_auc,
    },
    "full_pihd_reference": {
        "source": FULL_RESULT_PATH,
        "pooled_loso_auc": float(roc_auc_score(labels, full_probability)),
        "loso_prob": full_probability,
    },
    "paired_full_vs_control": comparison,
    "features": features,
    "labels": labels,
    "groups": groups,
    "elapsed_minutes": (time.time() - run_start) / 60.0,
}
np.save(OUTPUT_PATH, result, allow_pickle=True)

json_result = {
    "seed": SEED,
    "n_subjects": len(unique_subjects),
    "n_trials": len(labels),
    "control": result["control_definition"],
    "weight_matching": {
        "method": result["weight_matching"]["method"],
        "matched_fraction": float(np.mean(lambda_matched)),
        "lambda_median": float(np.median(lambda_values)),
        "lambda_iqr": [
            float(np.percentile(lambda_values, 25)),
            float(np.percentile(lambda_values, 75)),
        ],
        "match_epoch_range": [
            int(np.min(match_epochs)),
            int(np.max(match_epochs)),
        ],
        "maximum_relative_error": float(np.max(match_relative_errors)),
        "coefficient_clipping": False,
    },
    "smoothness_control": {
        "me_dz": me_dz,
        "mi_dz": mi_dz,
        "me_p_raw": float(me_p),
        "mi_p_raw": float(mi_p),
        "trial_cv_auc": cv_auc,
        "pooled_loso_auc": loso_auc,
        "pooled_loso_accuracy": loso_accuracy,
        "label_permutation_p": permutation_p,
    },
    "full_pihd": {
        "pooled_loso_auc": result["full_pihd_reference"]["pooled_loso_auc"]
    },
    "full_minus_smoothness": {
        "delta_auc": comparison["observed_delta_auc"],
        "ci95": comparison["cluster_bootstrap_ci95"].tolist(),
        "paired_swap_p_two_sided": comparison["paired_swap_p_two_sided"],
    },
}
with open(JSON_PATH, "w", encoding="utf-8") as stream:
    json.dump(json_result, stream, ensure_ascii=False, indent=2)

print("\n" + "=" * 78)
print(f"Matched Smoothness Control Results (seed={SEED})")
print("=" * 78)
print(f"ME d_z:                 {me_dz:+.3f} (p={me_p:.3g})")
print(f"MI d_z:                 {mi_dz:+.3f} (p={mi_p:.3g})")
print(f"Trial CV AUC:           {cv_auc:.3f}")
print(f"Pooled LOSO AUC:        {loso_auc:.3f}")
print(f"LOSO permutation p:     {permutation_p:.4f}")
print(
    f"Full PIHD AUC:          "
    f"{result['full_pihd_reference']['pooled_loso_auc']:.3f}"
)
print(f"Full - smooth Delta:    {comparison['observed_delta_auc']:+.3f}")
print(
    f"Cluster bootstrap CI:   "
    f"[{comparison['cluster_bootstrap_ci95'][0]:+.3f}, "
    f"{comparison['cluster_bootstrap_ci95'][1]:+.3f}]"
)
print(
    f"Paired swap p (2-sided): "
    f"{comparison['paired_swap_p_two_sided']:.4f}"
)
print(f"Lambda matched:         {np.mean(lambda_matched) * 100:.1f}% of trials")
print(f"Lambda median:          {np.median(lambda_values):.4g}")
print("=" * 78)
print(f"NPY saved:  {OUTPUT_PATH}")
print(f"JSON saved: {JSON_PATH}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print("Done.")
