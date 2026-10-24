# Initial estimation and evaluation

All 30 initial-model fits and 60 classifier fits are complete and verified.
Vaibhavi's classifier implementations and original results are unchanged.

## Setup and reproduce

Use Python 3.12.2 and the pinned dependencies. From `code/`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
python run_evaluation.py --out reproduced_results
```

Put the three provided datasets in `data/` first, or pass `--data-dir PATH`.
The downloaded workspace can use `--data-dir ../../datasets`.

The complete fresh command took **47.75 minutes** on the measured laptop:
ARM64 CPU, 8 logical cores, 8 GiB RAM, two workers,
two PyTorch threads per worker. This includes estimation, classifier training,
test evaluation, tables and plots. Installation and dataset setup are excluded.
Different hardware and system load can change the time.

## Settings

- 10 random 80/20 splits per dataset, seeds 0–9.
- 4 estimator epochs and 4 Forward epochs.
- Reweight-R: 2 warm-up + 4 fixed-matrix + 2 revision epochs.
- Training-only channel standardisation; no test labels used for model selection.
- Supplied FashionMNIST matrices; per-split CIFAR estimates.
- Standard deviations use `ddof=0`.

## Estimator

A noisy-label CNN is selected by noisy validation CE. Maximum-posterior training
anchors initialise T, following stage 1 of
[Xia et al. (2019)](https://proceedings.neurips.cc/paper/2019/hash/9308b0d6e5898366a4a986bc33f3d3e7-Abstract.html).
`T[i,j] = P(noisy=j | clean=i)` and noisy posterior is `p_clean @ T`.
The heuristic assumes sufficiently pure examples and compatible class ordering.
`--top-k 1` uses one anchor; larger values average candidates.

## Results

The workspace stores completed results in `evaluation_results/`. The distributed
ZIP labels them `reference_results/`, leaving the default output fresh.

```sh
python summarize_evaluation.py reference_results
```

Use `evaluation_results` instead when inspecting the working repository.
Each seed contains its initial/revised matrices, split indices, normalisation,
anchor indices, predictions and metrics. Top-level files contain accuracy and
matrix-error summaries, revision gains, tables, plots and measured runtime.
All runs, including weak CIFAR results, are retained. Revision changed no predictions
in the thirty paired evaluations; no accuracy benefit was observed under these settings.

Revision compares the selected checkpoint before revision with the selected
final checkpoint. Gains include extra training; they are not an equal-budget
fixed-matrix control. CIFAR's true matrix is unknown, so its estimation error
cannot be measured. Supplied FashionMNIST matrices start with zero matrix error.

The manifest checks data, source, settings and environment when resuming.
Choose a new `--out` after changing them. `--estimate-only` runs just estimation.
`--fashion-estimated --out sensitivity_results` adds an optional experiment
outside the measured default budget.

The report contribution and ZIP are outside the repository in `../../Deliverables/`
from `code/`. The group still needs its shared report sections and member details.
