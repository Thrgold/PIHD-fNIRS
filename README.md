# PIHD: Physics-Informed Hemodynamic Decomposition for fNIRS

Research code and archived results for *Physics-Informed Hemodynamic Decomposition
for fNIRS Motor Task Classification*. This repository accompanies a manuscript;
it does not claim conference acceptance.

PIHD is an offline, paradigm-informed, two-stage decomposition model with
Balloon–Windkessel residual constraints. Task timing informs optimization.
Positive task–rest contrasts do not independently establish neural specificity,
and the observed LOSO AUC advantage over Wavelet was not statistically significant.

The analyzed dataset is SEFMID (Science Data Bank,
[doi:10.57760/sciencedb.30795](https://doi.org/10.57760/sciencedb.30795)).

## Layout

```text
src/       numbered experiment and plotting scripts (original names retained)
result/    numbered, immutable experiment archives; three seeds where available
figures/   final author-selected PDF figures
docs/      experiment index, numerical summary, and file checksums
results/   ignored working copy, restored by src/00_prepare_workspace.py
```

The singular `result/` is the public archive. Historical scripts still read/write
`results/`; the preparation step reconstructs that layout to preserve the actual
experiment implementations. New runs do **not** silently change archived results.

## Setup and use

Use Python 3.10 or later. Install a PyTorch build appropriate for your CPU/CUDA
environment, then install the dependencies:

```sh
python -m pip install -r requirements.txt
python src/00_prepare_workspace.py
```

Obtain SEFMID from the Science Data Bank using the DOI above and prepare the
required participant-level inputs locally. This repository does not redistribute
the dataset or derived participant-level recordings. Local data should be placed
under `data/`, which is excluded from Git tracking.

Each seed-specific experiment accepts `42`, `123`, or `999` as its first argument:

```sh
python src/7_loso_classification_9methods_48subj.py 42
python src/8.5_loso_permutation_test.py 42
python src/8.6_activation_task_rest_sensitivity_9methods_48subj.py 42
```

Repeat required experiments for the other two seeds. Numbering identifies stages,
not an instruction to run every script. Archived results are already included:
no retraining is needed merely to inspect them. See [the experiment index](docs/EXPERIMENTS.md)
for dependencies and optional scripts. Many experiments optimize per trial and
can take hours; simultaneous GPU jobs may exhaust resources.

NumPy object files require `allow_pickle=True`; load only files from trusted sources.
Checksums identify the packaged files, not an independently audited experiment.

## Results and figures

The numerical summary in [docs/RESULTS.md](docs/RESULTS.md) is derived from the
archived outputs and states the evidential limits explicitly. Run
`python src/00.2_validate_results.py` for read-only cross-file checks.

The four PDFs in `figures/` are the author-selected manuscript figures. Fig. 2
was manually curated. Scripts 13 and 13.5 regenerate data-derived working versions
of Figs. 3–4 but do not overwrite Fig. 2; regenerated files need not be
pixel-identical to editorially adjusted final artwork.

## Data and license

Code retains the existing MIT license. This license does **not** grant rights to
the SEFMID data. Dataset redistribution terms must be checked separately before
using or sharing participant-level records. The entire local `data/` directory is
ignored to prevent accidental publication.

Repository organization did not rerun or alter the reported experiments.
