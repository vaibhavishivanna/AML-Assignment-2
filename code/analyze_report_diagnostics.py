"""Post-hoc diagnostics from saved matrices and clean-test predictions only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from experiments.data_utils import default_data_dir
from transition_matrix import KNOWN_T


def analyze(results, data_dir, output):
    results, data_dir, output = Path(results), Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = {'known_matrix_feasibility': {}, 'CIFAR': {}}
    for dataset, matrix in KNOWN_T.items():
        values = np.stack([np.load(p / 'T_estimated.npy') for p in sorted((results / dataset).glob('seed_*'))])
        implied_clean = values @ np.linalg.inv(matrix)
        diagonal = np.diagonal(values, axis1=1, axis2=2)
        evidence['known_matrix_feasibility'][dataset] = {
            'n_rows': len(values) * 3, 'diagonal_min': float(diagonal.min()), 'diagonal_max': float(diagonal.max()),
            'diagonal_above_true_posterior_upper_bound': int((diagonal > matrix.max(0) + 1e-7).sum()),
            'rows_outside_true_posterior_convex_hull': int((implied_clean.min(-1) < -1e-6).sum()),
            'minimum_implied_clean_probability': float(implied_clean.min())}
    with np.load(data_dir / 'CIFAR.npz') as source:
        labels = source['Yts']
    counts = np.bincount(labels, minlength=3)
    rows = []
    for folder in sorted((results / 'CIFAR').glob('seed_*'), key=lambda p: int(p.name.removeprefix('seed_'))):
        row = {'seed': int(folder.name.removeprefix('seed_')),
               'estimated_condition_number': float(np.linalg.cond(np.load(folder / 'T_estimated.npy')))}
        for method in ('Forward', 'Reweight-R'):
            predictions = np.load(folder / f'estimated_{method}_predictions.npy')
            confusion = np.zeros((3, 3), dtype=int)
            np.add.at(confusion, (labels, predictions), 1)
            row[method] = {'accuracy': float(100 * np.mean(predictions == labels)),
                           'recall_percent': (100 * confusion.diagonal() / counts).tolist(),
                           'confusion_counts': confusion.tolist()}
        rows.append(row)
    evidence['CIFAR']['test_class_counts'] = counts.tolist()
    evidence['CIFAR']['per_seed'] = rows
    evidence['CIFAR']['summary'] = {}
    for method in ('Forward', 'Reweight-R'):
        recalls = np.array([row[method]['recall_percent'] for row in rows])
        accuracies = np.array([row[method]['accuracy'] for row in rows])
        conditions = np.array([row['estimated_condition_number'] for row in rows])
        evidence['CIFAR']['summary'][method] = {
            'recall_mean_percent': recalls.mean(0).tolist(), 'recall_std_percent': recalls.std(0).tolist(),
            'accuracy_min': float(accuracies.min()), 'accuracy_max': float(accuracies.max()),
            'condition_accuracy_pearson_r': float(pearsonr(conditions, accuracies).statistic),
            'condition_accuracy_spearman_r': float(spearmanr(conditions, accuracies).statistic)}
    evidence['limitations'] = ('Clean test labels are used for post-hoc evaluation only. Correlations are exploratory, '
                              'not causal. Feasibility tests use supplied T and identify posterior inconsistency, '
                              'not observed clean training-label errors.')
    (output / 'saved_diagnostics.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence['known_matrix_feasibility'], indent=2))
    print(json.dumps(evidence['CIFAR']['summary'], indent=2))
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.results, args.data_dir, args.out)
