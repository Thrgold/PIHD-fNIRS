# -*- coding: utf-8 -*-
"""Benchmark the complete per-trial PIHD optimization used in the paper.

This is deliberately different from ``12_measure_inference_time.py``:
the old script times only trained-network forward passes, whereas the current
PIHD implementation optimizes a new model for every complete trial.  This
script therefore reports the end-to-end offline trial-processing time,
including interpolation, normalization, Stage 1 (400 epochs), Stage 2
(800 epochs), and output generation.

To prevent implementation drift, the network classes and
``train_decomposition_v4`` are extracted from
``2_real_data_decomposition.py`` without executing that script's main body.

Usage
-----
python 12.5_measure_full_optimization_time.py
python 12.5_measure_full_optimization_time.py --n-trials 30 --n-warmup 2
python 12.5_measure_full_optimization_time.py --seed 42 --device cuda

Outputs
-------
results/npy/runtime_full_optimization_seed{seed}_n{n}.npy
results/json/runtime_full_optimization_seed{seed}_n{n}.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import random
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
import torch
import torch.nn as nn
from scipy import stats
from scipy.special import gamma as gamma_fn

# Prevent Windows terminals from failing when the project path contains Chinese text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_SCRIPT = os.path.join(SCRIPT_DIR, "2_real_data_decomposition.py")
DATA_PATH = os.path.join(REPO_ROOT, "data", "eeg_features_all_v6.npy")
NPY_DIR = os.path.join(REPO_ROOT, "results", "npy")
JSON_DIR = os.path.join(REPO_ROOT, "results", "json")

N_SUBJECTS = 48
N_POINTS = 80
ALPHA = 0.32
E0 = 0.34
SAFE = 1e-6
N_EPOCHS_STAGE1 = 400
N_EPOCHS_STAGE2 = 800
LR = 2e-3
SYS_FREQS = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 1.00]

EXTRACT_NAMES = {
    "set_seed",
    "FVQNet",
    "FinNet",
    "SystemicNoise",
    "make_hrf_prior",
    "bw_residual",
    "train_decomposition_v4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure complete offline PIHD optimization time per trial."
    )
    parser.add_argument("--seed", type=int, default=42, help="PIHD random seed.")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=120527,
        help="Seed used only for objective benchmark-trial sampling.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of measured trials (default: 30).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=2,
        help="Number of full warm-up trials excluded from statistics (default: 2).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Execution device (default: auto).",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return requested


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def load_exact_main_implementation(seed: int, device: str, t_ds: np.ndarray):
    """Load selected definitions from Step 2 without running its experiment."""
    with open(SOURCE_SCRIPT, "r", encoding="utf-8-sig") as stream:
        tree = ast.parse(stream.read(), filename=SOURCE_SCRIPT)

    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in EXTRACT_NAMES:
            selected.append(node)
            found.add(node.name)

    missing = EXTRACT_NAMES - found
    if missing:
        raise RuntimeError(f"Definitions missing from Step 2: {sorted(missing)}")

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "torch": torch,
        "nn": nn,
        "random": random,
        "gamma_fn": gamma_fn,
        "SEED": seed,
        "DEVICE": device,
        "ALPHA": ALPHA,
        "E0": E0,
        "SAFE": SAFE,
        "N_EPOCHS_STAGE1": N_EPOCHS_STAGE1,
        "N_EPOCHS_STAGE2": N_EPOCHS_STAGE2,
        "LR": LR,
        "SYS_FREQS": SYS_FREQS,
        # Step 2 currently refers to this time grid when constructing its output.
        "t_ds": t_ds,
    }
    exec(compile(module, SOURCE_SCRIPT, "exec"), namespace)
    return namespace["set_seed"], namespace["train_decomposition_v4"]


def select_main_cohort(trials: list, seed: int) -> list:
    """Reproduce Step 2's seeded selection of 48 participants."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    subjects = sorted({trial["subject"] for trial in trials})
    if len(subjects) > N_SUBJECTS:
        selected_subjects = list(np.random.choice(subjects, N_SUBJECTS, replace=False))
    else:
        selected_subjects = subjects
    selected = [trial for trial in trials if trial["subject"] in selected_subjects]
    if not selected:
        raise RuntimeError("No trials remained after participant selection.")
    return selected


def prepare_trial(trial: dict, t_ds: np.ndarray) -> np.ndarray:
    hbo_ds = np.interp(t_ds, trial["t"], trial["hbo_n"])
    rest_mask = t_ds < 8
    baseline_mean = float(np.mean(hbo_ds[rest_mask]))
    baseline_std = float(np.std(hbo_ds[rest_mask]) + 1e-10)
    return (hbo_ds - baseline_mean) / baseline_std


def hardware_metadata(device: str) -> dict:
    meta = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "requested_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        meta.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_gib": props.total_memory / (1024**3),
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return meta


def summarize_seconds(values: np.ndarray) -> dict:
    n = values.size
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    if n > 1:
        q = float(stats.t.ppf(0.975, df=n - 1))
        ci = [mean - q * sem, mean + q * sem]
    else:
        ci = [mean, mean]
    return {
        "n": int(n),
        "mean": mean,
        "sd": sd,
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean_ci95_t": [float(ci[0]), float(ci[1])],
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    args = parse_args()
    if args.n_trials < 2:
        raise ValueError("Use at least two measured trials for a standard deviation.")
    if args.n_warmup < 0:
        raise ValueError("--n-warmup must be nonnegative.")
    if not os.path.isfile(SOURCE_SCRIPT):
        raise FileNotFoundError(SOURCE_SCRIPT)
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    device = resolve_device(args.device)
    os.makedirs(NPY_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    t_ds = np.linspace(0, 32, N_POINTS)
    set_seed, train_decomposition = load_exact_main_implementation(
        args.seed, device, t_ds
    )
    set_seed(args.seed)

    data = np.load(DATA_PATH, allow_pickle=True).item()
    cohort = select_main_cohort(data["trials"], args.seed)
    required = args.n_trials + args.n_warmup
    if required > len(cohort):
        raise ValueError(f"Requested {required} trials but the cohort has {len(cohort)}.")

    sampling_rng = np.random.default_rng(args.sample_seed)
    sampled_indices = sampling_rng.choice(len(cohort), size=required, replace=False)
    warmup_indices = sampled_indices[: args.n_warmup]
    measured_indices = sampled_indices[args.n_warmup :]

    print("=" * 78)
    print("Complete offline PIHD per-trial optimization benchmark")
    print("=" * 78)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Configuration: Stage 1={N_EPOCHS_STAGE1}, Stage 2={N_EPOCHS_STAGE2}")
    print(f"Warm-up trials: {args.n_warmup}; measured trials: {args.n_trials}")
    print("Timing includes interpolation, normalization, both optimization stages, and output generation.")

    for position, index in enumerate(warmup_indices, start=1):
        trial = cohort[int(index)]
        hbo_norm = prepare_trial(trial, t_ds)
        synchronize(device)
        started = time.perf_counter()
        train_decomposition(t_ds, hbo_norm, device=device)
        synchronize(device)
        elapsed = time.perf_counter() - started
        print(f"Warm-up {position}/{args.n_warmup}: {elapsed:.3f} s", flush=True)

    records = []
    for position, index in enumerate(measured_indices, start=1):
        trial = cohort[int(index)]
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        synchronize(device)
        total_started = time.perf_counter()
        hbo_norm = prepare_trial(trial, t_ds)
        synchronize(device)
        optimization_started = time.perf_counter()
        train_decomposition(t_ds, hbo_norm, device=device)
        synchronize(device)
        optimization_seconds = time.perf_counter() - optimization_started
        total_seconds = time.perf_counter() - total_started

        record = {
            "measurement_number": position,
            "cohort_trial_index": int(index),
            "subject_id": str(trial.get("subject", "")),
            "hand": str(trial.get("hand", "")),
            "optimization_seconds": float(optimization_seconds),
            "end_to_end_seconds": float(total_seconds),
            "peak_gpu_memory_mib": (
                float(torch.cuda.max_memory_allocated() / (1024**2))
                if device == "cuda"
                else None
            ),
        }
        records.append(record)
        print(
            f"[{position:02d}/{args.n_trials}] subject={record['subject_id']} "
            f"hand={record['hand']} total={total_seconds:.3f} s",
            flush=True,
        )

    optimization_times = np.asarray(
        [record["optimization_seconds"] for record in records], dtype=float
    )
    end_to_end_times = np.asarray(
        [record["end_to_end_seconds"] for record in records], dtype=float
    )
    peak_memory = np.asarray(
        [record["peak_gpu_memory_mib"] for record in records if record["peak_gpu_memory_mib"] is not None],
        dtype=float,
    )

    result = {
        "benchmark_name": "complete_offline_pihd_per_trial_optimization",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Complete noncausal offline trial-specific optimization time; "
            "this is not trained-network forward inference latency."
        ),
        "source_implementation": os.path.basename(SOURCE_SCRIPT),
        "source_implementation_path": SOURCE_SCRIPT,
        "configuration": {
            "seed": args.seed,
            "sample_seed": args.sample_seed,
            "n_subjects": N_SUBJECTS,
            "cohort_trials": len(cohort),
            "n_points": N_POINTS,
            "stage1_epochs": N_EPOCHS_STAGE1,
            "stage2_epochs": N_EPOCHS_STAGE2,
            "learning_rate": LR,
            "n_warmup": args.n_warmup,
            "n_measured": args.n_trials,
        },
        "hardware": hardware_metadata(device),
        "records": records,
        "optimization_seconds": summarize_seconds(optimization_times),
        "end_to_end_seconds": summarize_seconds(end_to_end_times),
        "peak_gpu_memory_mib": (
            summarize_seconds(peak_memory) if peak_memory.size else None
        ),
    }

    stem = f"runtime_full_optimization_seed{args.seed}_n{args.n_trials}"
    npy_path = os.path.join(NPY_DIR, stem + ".npy")
    json_path = os.path.join(JSON_DIR, stem + ".json")
    np.save(npy_path, result, allow_pickle=True)
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(json_ready(result), stream, ensure_ascii=False, indent=2)

    summary = result["end_to_end_seconds"]
    print("\n" + "=" * 78)
    print("End-to-end time per trial")
    print("=" * 78)
    print(f"Mean +/- SD: {summary['mean']:.3f} +/- {summary['sd']:.3f} s")
    print(f"Median [IQR]: {summary['median']:.3f} "
          f"[{summary['q1']:.3f}, {summary['q3']:.3f}] s")
    print(f"95% CI for mean: [{summary['mean_ci95_t'][0]:.3f}, "
          f"{summary['mean_ci95_t'][1]:.3f}] s")
    if result["peak_gpu_memory_mib"] is not None:
        memory = result["peak_gpu_memory_mib"]
        print(f"Peak allocated GPU memory: {memory['mean']:.1f} +/- {memory['sd']:.1f} MiB")
    print(f"NPY saved:  {npy_path}")
    print(f"JSON saved: {json_path}")


if __name__ == "__main__":
    main()
