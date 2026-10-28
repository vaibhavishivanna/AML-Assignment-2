# Transition estimation and evaluation

## Setup and reproduce

Use Python 3.12 and the pinned dependencies. From `code/`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python run_evaluation.py --runs 10 --workers 4 --threads 1 --out reproduced_results
python verify_evaluation.py --results reproduced_results
```

Place the three assignment datasets in `code/data/`. The default is `data/`
relative to the Python scripts, not the launch working directory. Keep `.gitkeep`
tracked and exclude the `.npz` files from Git and the submission ZIP. Alternatively
add `--data-dir PATH` to both commands for another relative location. The command
above uses the tested fast profile: four workers and one thread each. The original
default of two workers/two threads remains available for lower-memory machines. Use a fresh output path: manifests
check source, data, settings, environment and the transition convention when resuming.

## Matrix contract and estimator

`T[clean, noisy] = P(noisy | clean)`; rows sum to one. The noisy posterior is
`p_clean @ T`. Supplied FashionMNIST matrices already have this orientation.
The lecture matrix is its transpose; no transpose is needed for this implementation.
For FashionMNIST0.3, the first row `[0.7, 0.3, 0]` describes noisy outcomes of
clean class 0. CIFAR estimates, revision, error metrics and heatmaps follow the same
contract. `transition_matrix.py` provides the shared validation and known matrices.

A preliminary noisy-label CNN is selected by noisy-validation cross-entropy.
For each noisy class index, the estimator averages the full predicted posteriors
of the top-k training examples. Those vectors become clean-class rows under the
anchor assumptions: sufficiently pure examples, compatible class ordering and
reliable posteriors. Averaging does not guarantee these assumptions or identify
CIFAR's true matrix. No test labels select anchors. `--top-k 1` is the single-anchor
variant; the current default is ten.

Both robust classifiers explicitly use the matrix, a choice allowed by the supplied
staff clarification. Classifier entry checks reject invalid matrices. Revision
clips negative entries and normalizes rows; a fully clipped row falls back to its
input row so it remains a probability distribution.

## Training and evaluation

- Ten random 80/20 splits per dataset, seeds 0-9, shared by both methods.
- Four estimator and four Forward epochs.
- Reweight-R: two warm-up + four fixed-matrix + eight revision epochs.
- Reweight-R uses detached importance weights in its fixed stage and Forward loss
  during revision; it is a modified T-Revision-inspired hybrid.
- Training-only channel standardization and noisy-validation model selection.
- Supplied matrices for FashionMNIST; per-split estimated matrices for CIFAR.
- Mean clean test accuracy and population SD (`ddof=0`).

Revision compares the selected fixed-stage checkpoint with the selected final
checkpoint. It includes extra optimization, so gains do not isolate matrix revision
against an equal-budget control. CIFAR has no known matrix-error target.

## Verified faster execution profile

The complete fresh workload took **58 minutes 33.20 seconds** with
`--workers 4 --threads 1`, compared with **70 minutes 0.19 seconds** for the original
`--workers 2 --threads 2` profile on this eight-logical-CPU ARM64 Mac. This saves
**16.36%** of measured elapsed time, including startup, all fits, evaluation,
tables and plots. Installation, copying datasets and subsequent verification are
excluded. Runtime is hardware- and load-dependent.

All **360 saved arrays** matched the 70-minute run exactly, and all 60 final plus
30 pre-revision accuracies were independently recomputed. Training/evaluation
source and dataset hashes match the baseline. No data, fits, epochs, validation,
transition estimation or model-selection steps were removed or changed.

See `runtime_optimization_results.json` for the trial comparison. The full local
outputs are `runtime_optimization_20261029/full_mixed/w4_t1/`. These are fresh
verified results, separate from the historical shared-report records below:

| Dataset | Forward | Reweight-R |
|---|---:|---:|
| FashionMNIST0.3 | 98.51 +/- 0.11 | 98.80 +/- 0.13 |
| FashionMNIST0.6 | 95.05 +/- 0.97 | 95.21 +/- 0.66 |
| CIFAR | 64.72 +/- 8.90 | 65.68 +/- 7.55 |

Use a fresh output folder after changing thread settings. The original profile and
its measured record (`runtime_benchmark.json`) are preserved. The shared report
has not been edited during this code/runtime task.

## Saved results and provenance

The final report uses `final_evaluation/`:

```sh
python summarize_evaluation.py final_evaluation
python verify_evaluation.py --results final_evaluation
```

These are historical records consolidated from two experiments without retraining:
known-matrix FashionMNIST from `revision_forward_10seeds/`, and CIFAR with
`top_k=10` from `cifar_k10_evaluation/`. Both use eight revision epochs.
FashionMNIST classifier accuracy does not depend on its auxiliary estimated matrix.
Its saved per-seed initial estimates and `estimator_error_summary.csv` use `k=1`.
Separate `estimator_k10_*` CSVs describe the ten-anchor FashionMNIST estimator study.
The latter are not regenerated from the per-seed k=1 matrices by the summarizer.
A fresh current-default run estimates all three datasets with k=10; its FashionMNIST
estimator errors should be compared with that separate k=10 study.

Historical `final_evaluation/` accuracy (percent, mean +/- population SD):

| Dataset | Forward | Reweight-R |
|---|---:|---:|
| FashionMNIST0.3 | 98.52 +/- 0.11 | 98.78 +/- 0.14 |
| FashionMNIST0.6 | 95.13 +/- 0.95 | 95.25 +/- 0.67 |
| CIFAR | 66.32 +/- 7.17 | 65.85 +/- 6.14 |

`evaluation_results/` contains the older k=1/two-revision-epoch study;
`final_results/` contains the original classifier study. The earlier 47.75-minute
full-run timing belongs to the older profile. The original two-worker full profile was freshly measured on 29 October 2026 at **4,200.19
seconds (1 hour 10 minutes)**, including process startup, all model fits, evaluation,
tables and plots. It used an ARM64 Mac with eight logical CPU cores, two workers
and two threads per worker. Installation, dataset copying and independent
verification are excluded. `runtime_benchmark.json` records the timing, settings,
source/data hashes and fresh accuracy summary. All 60 final and 30 pre-revision
accuracies were independently recomputed successfully. The local full run is
`runtime_evaluation_20261029/` (ignored by Git); its summaries differ from the
historical `final_evaluation/` records, so these remain separate. A source change requires a fresh output directory; existing
historical results remain available for read-only verification.

Verification checks splits, stochastic initial/input/revised matrices, equality of
classifier inputs to the supplied or per-split estimated matrices, matrix error
records, revision gains and accuracy summaries. With datasets available, it also
recomputes final and pre-revision accuracies from saved predictions. Passing these
checks establishes internal consistency, not the unknown CIFAR matrix's accuracy.

`--estimate-only` runs estimation alone. `--fashion-estimated --out sensitivity_results`
adds an optional sensitivity experiment using estimated FashionMNIST matrices.

Each invocation prints its measured elapsed time and saves `runtime_measured.json`.
That timer covers setup through tables/plots, after Python imports. For a resumed
run it measures only that invocation; `completed_splits_already_present_at_start`
records how many full split results were already cached.
