# Final Evaluation Results

This directory contains the selected final classification results for Assignment 2.

## Experimental Setup

- **Datasets:** FashionMNIST0.3, FashionMNIST0.6, CIFAR
- **Methods:** Forward Correction and Reweight-R
- **Runs:** 10 independent runs per dataset and method
- **Train/validation split:** 80/20
- **Metric:** Clean test accuracy (%), reported as mean ± population standard deviation (`ddof=0`)
- **Revision stage:** Forward-corrected loss, 8 epochs

## Result Provenance

| Dataset | Original experiment directory | Classification transition matrix |
|---|---|---|
| FashionMNIST0.3 | `revision_forward_10seeds/` | Known matrix |
| FashionMNIST0.6 | `revision_forward_10seeds/` | Known matrix |
| CIFAR | `cifar_k10_evaluation/` | Estimated matrix, `top_k=10` |

The dataset experiment folders were copied without retraining. Summary tables and figures were regenerated using `summarize_evaluation.py`.

## Transition Matrix Estimator

Two estimator analyses are retained:

- `estimator_error_summary.csv`: results associated with the original FashionMNIST classification experiments (`top_k=1`).
- `estimator_k10_error_summary.csv`: separately evaluated estimator using `top_k=10`, selected after sensitivity testing.

The corresponding per-run CSV files are also included.

## Verification

The consolidated results were verified using:

```bash
python verify_evaluation.py --results final_evaluation
```

Verification passed for 30 train/validation splits, 30 estimated transition matrices, 60 prediction files, 6 accuracy summaries and 60/60 independently recomputed clean-label test accuracies.

## Regenerating Summary Tables and Figures

From the `code/` directory:

```bash
python summarize_evaluation.py final_evaluation
```

This regenerates the main summaries and figures from the saved experiment records. The separately copied `estimator_k10_*` CSV files represent a distinct estimator study and should be preserved independently.

The original dataset files are not included in this directory.