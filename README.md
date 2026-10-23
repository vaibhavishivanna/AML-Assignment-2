# AML-Assignment-2
# Assignment 2 – label-noise robust classifiers + data preparation (Group 39)

Scope of this folder: **data preparation** and the **classification models** (CE baseline, Forward,
Backward, Reweighting + T-Revision). The transition-matrix estimator is a separate part; its output
(3x3 `.npy`) is the only thing needed here to run CIFAR.

Requirements: Python 3.9+, numpy, torch (CPU is enough), matplotlib; pytest for the tests.

## Layout
    algorithm/          networks.py, losses.py, base.py (shared training loop + model selection)
                        ce_baseline.py, forward_correction.py, backward_correction.py,
                        reweight_t_revision.py
    data/               EMPTY in the submission - copy FashionMNIST0.3.npz, FashionMNIST0.6.npz, CIFAR.npz here
    experiments/        data_utils.py (loading, preprocessing, 80/20 split, given T),
                        visualize.py (data figures), run_demo.py (1-minute demo)
    final_results/      summary.csv, accuracy_summary_table.tex, raw_runs.json, run_log.txt, figures/
    tests/              unit tests (losses, data prep, classifiers)
    run_experiments.py  full experiment: 10 random 80/20 splits x methods x datasets (resumable)
    evaluation.py       top-1 accuracy, mean +- std, CSV / LaTeX table
    plot_results.py     accuracy comparison figure
    demo_algorithms.py  numeric check that backward correction is unbiased

## How to run
    pip install numpy torch matplotlib pytest
    python -m pytest -q tests                       # unit tests
    python demo_algorithms.py                       # loss sanity check
    python -m experiments.visualize                 # data figures -> final_results/figures
    python -m experiments.run_demo data/FashionMNIST0.3.npz     # quick demo (~1 min)

    python run_experiments.py                       # FashionMNIST0.3 + 0.6, given T (about 40 min per dataset on 2 CPU cores)
    python run_experiments.py --datasets CIFAR --T_file estimated_T.npy    # CIFAR with the estimated T
    python plot_results.py                          # figures/accuracy_comparison.png

`run_experiments.py` resumes from `final_results/raw_runs.json`, so an interrupted run can be restarted.

## Preprocessing
pixels /255 -> float32; FashionMNIST reshaped to 1x28x28, CIFAR to 3x32x32; per-channel standardisation
using statistics of the training split only; no augmentation; labels used as given.
The 80/20 train/validation split is drawn independently per run (seeds 0-9).

## Protocol
All classifiers use the same CNN. Model selection uses only the noisy validation split (forward-corrected
NLL under the supplied T). The clean test set is used only for the final top-1 accuracy.
