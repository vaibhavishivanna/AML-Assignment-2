"""Verify saved Assignment 2 experiment results, optionally against clean test labels.

Run from code/: python verify_evaluation.py
Optional: python verify_evaluation.py --data-dir data
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED = {'FashionMNIST0.3': 18000, 'FashionMNIST0.6': 18000, 'CIFAR': 15000}


def verify(root, data_dir):
    summary_file = root / 'accuracy_summary.csv'
    if not summary_file.exists():
        raise FileNotFoundError(f'Missing {summary_file}. Run summarize_evaluation.py first.')
    with summary_file.open(newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 6, f'Expected six summary rows, got {len(rows)}'
    checked_labels = 0
    for dataset, n in EXPECTED.items():
        source = data_dir / f'{dataset}.npz'
        labels = None
        if source.is_file():
            with np.load(source) as data:
                labels = np.asarray(data['Yts'])
            assert labels.shape == (3000,), (dataset, labels.shape)
            assert np.all(np.isin(labels, [0, 1, 2]))
        else:
            print(f'NOTE: clean labels unavailable for {dataset}: {source}')
        for seed in range(10):
            folder = root / dataset / f'seed_{seed:02d}'
            with np.load(folder / 'split_indices.npz') as split:
                train, valid = split['train'], split['validation']
            assert len(train) == int(.8 * n) and len(valid) == n - int(.8 * n), (dataset, seed)
            assert np.array_equal(np.sort(np.r_[train, valid]), np.arange(n)), (dataset, seed)
            assert len(np.unique(train)) == len(train) and len(np.unique(valid)) == len(valid)
            T = np.load(folder / 'T_estimated.npy')
            assert T.shape == (3, 3) and np.isfinite(T).all()
            assert np.all(T >= -1e-7) and np.allclose(T.sum(axis=1), 1, atol=1e-5)
            mode = 'estimated' if dataset == 'CIFAR' else 'known'
            for method in ('Forward', 'Reweight-R'):
                stem = folder / f'{mode}_{method}'
                record = json.loads(stem.with_suffix('.json').read_text())
                predictions = np.load(str(stem) + '_predictions.npy')
                assert predictions.shape == (3000,) and np.all(np.isin(predictions, [0, 1, 2]))
                assert record['n_test'] == len(predictions)
                if labels is not None:
                    recalculated = 100 * np.mean(predictions == labels)
                    assert np.isclose(recalculated, record['accuracy'], atol=1e-8), (
                        dataset, seed, method, recalculated, record['accuracy'])
                    checked_labels += 1
    for row in rows:
        dataset, mode, method = row['dataset'], row['mode'], row['method']
        values = np.array([
            json.loads((root / dataset / f'seed_{seed:02d}' / f'{mode}_{method}.json').read_text())['accuracy']
            for seed in range(10)
        ])
        assert int(row['n_runs']) == 10
        assert np.isclose(float(row['accuracy_mean']), values.mean(), atol=1e-10)
        assert np.isclose(float(row['accuracy_std']), values.std(ddof=0), atol=1e-10)
        print(f'{dataset:17s} {method:10s} {values.mean():6.2f} ± {values.std(ddof=0):5.2f}% (10 runs)')
    print('PASS: 30 splits, 30 estimated matrices, 60 predictions, 6 summaries.')
    print(f'PASS: Recomputed {checked_labels}/60 test accuracies using clean labels.' if checked_labels else
          'LIMITATION: No original .npz datasets found, so test-label accuracy was not independently recomputed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument('--results', type=Path, default=here / 'evaluation_results')
    parser.add_argument('--data-dir', type=Path, default=here / 'data')
    args = parser.parse_args()
    verify(args.results, args.data_dir)