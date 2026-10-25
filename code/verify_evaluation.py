"""Verify the integrity of saved evaluation outputs.
Does not require the original (excluded) datasets; cannot re-check correctness against Yts.
Run: python verify_evaluation.py
"""
from pathlib import Path
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parent / 'evaluation_results'
DATASETS = {'FashionMNIST0.3': 18000, 'FashionMNIST0.6': 18000, 'CIFAR': 15000}
rows = list(csv.DictReader((ROOT / 'accuracy_summary.csv').open()))
assert len(rows) == 6, f'Expected 6 summary rows, got {len(rows)}'
for dataset, n in DATASETS.items():
    for seed in range(10):
        directory = ROOT / dataset / f'seed_{seed:02d}'
        assert directory.is_dir(), directory
        split = np.load(directory / 'split_indices.npz')
        train, valid = split['train'], split['validation']
        assert len(train) == int(.8*n) and len(valid) == n - int(.8*n)
        assert np.array_equal(np.sort(np.r_[train,valid]), np.arange(n)), (dataset, seed)
        assert len(np.unique(train)) == len(train) and len(np.unique(valid)) == len(valid)
        T = np.load(directory / 'T_estimated.npy')
        assert T.shape == (3,3) and np.all(T >= -1e-7) and np.allclose(T.sum(axis=1), 1, atol=1e-5)
        mode = 'estimated' if dataset == 'CIFAR' else 'known'
        for method in ('Forward','Reweight-R'):
            stem = directory / f'{mode}_{method}'
            record = json.loads(stem.with_suffix('.json').read_text())
            predictions = np.load(str(stem) + '_predictions.npy')
            assert predictions.shape == (3000,) and np.all(np.isin(predictions, [0,1,2]))
            assert record['n_test'] == 3000
for row in rows:
    dataset, mode, method = row['dataset'], row['mode'], row['method']
    values = np.array([json.loads((ROOT / dataset / f'seed_{seed:02d}' / f'{mode}_{method}.json').read_text())['accuracy'] for seed in range(10)])
    assert int(row['n_runs']) == 10
    assert np.isclose(float(row['accuracy_mean']), values.mean(), atol=1e-10)
    assert np.isclose(float(row['accuracy_std']), values.std(ddof=0), atol=1e-10)
    print(f'{dataset:17s} {method:10s} {values.mean():6.2f} ± {values.std(ddof=0):5.2f}% (10 runs)')
print('PASS: 30 complete 80/20 splits, 30 valid estimated matrices, 60 prediction files, 6 verified summaries.')
print('LIMITATION: Ground-truth test-label accuracy cannot be recomputed without the original .npz datasets.')