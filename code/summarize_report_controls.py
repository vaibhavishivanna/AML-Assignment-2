"""Verify and summarize the full supplementary study; generate report tables/figures."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from paired_analysis import paired_statistics, holm_adjust
from analyze_report_diagnostics import analyze
from experiments.data_utils import default_data_dir
from transition_matrix import KNOWN_T, matrix_errors

DATASETS = ('FashionMNIST0.3', 'FashionMNIST0.6', 'CIFAR')
SHORT = {'FashionMNIST0.3': 'FM0.3', 'FashionMNIST0.6': 'FM0.6', 'CIFAR': 'CIFAR'}


def pm(values, decimals=2):
    values = np.asarray(values)
    return f'${values.mean():.{decimals}f}\\pm{values.std(ddof=0):.{decimals}f}$'


def tabular(path, headers, rows):
    text = ['\\begin{tabular}{l' + 'c' * (len(headers) - 1) + '}', '\\toprule',
            ' & '.join(headers) + r'\\', '\\midrule']
    text += [' & '.join(map(str, row)) + r'\\' for row in rows]
    Path(path).write_text('\n'.join(text + ['\\bottomrule', '\\end{tabular}']) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(baseline, controls, data_dir, output, assets):
    baseline, controls, data_dir, output, assets = map(Path, (baseline, controls, data_dir, output, assets))
    config = json.loads((controls / 'study_manifest.json').read_text())
    source_root = Path(__file__).resolve().parent
    expected_hashes = {source_root / name: digest for name, digest in config['source_sha256'].items()}
    expected_hashes[source_root / 'run_report_controls.py'] = config['study_source_sha256']
    expected_hashes[baseline / 'manifest.json'] = config['baseline_manifest_sha256']
    expected_hashes.update({data_dir / f'{name}.npz': digest for name, digest in config['data_sha256'].items()})
    for path, expected in expected_hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Study source/data provenance mismatch: {path}')
    runs = config['runs']
    if runs != 10:
        raise ValueError('Final report requires all ten controlled repetitions per dataset')
    for dataset in DATASETS:
        for seed in range(runs):
            if not (controls / dataset / f'seed_{seed:02d}' / 'complete.json').exists():
                raise ValueError(f'Incomplete study: {dataset} seed={seed}')
    if len(list((controls / 'cifar_controls').glob('seed_*/complete.json'))) != runs:
        raise ValueError('Incomplete CIFAR randomness controls')
    output.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    saved = analyze(baseline, data_dir, output)
    values, histories, calibration, sensitivity, contrasts = {}, {}, {}, [], []
    revision_diagnostics = {}
    evidence_paths = [controls / 'study_manifest.json', baseline / 'manifest.json']
    for dataset in DATASETS:
        with np.load(data_dir / f'{dataset}.npz') as source:
            labels = source['Yts']
        values[dataset] = {name: [] for name in ('CE4', 'Forward4', 'Forward14', 'Fixed14', 'Reweight14')}
        histories[dataset], calibration[dataset] = [], []
        revision_diagnostics[dataset] = []
        mode = 'estimated' if dataset == 'CIFAR' else 'known'
        matrices_by_variant = {str(k): [] for k in (1, 5, 10, 20)}
        matrices_by_variant['calibrated10'] = []
        for seed in range(runs):
            folder = controls / dataset / f'seed_{seed:02d}'
            reference = baseline / dataset / f'seed_{seed:02d}'
            for name in ('CE4', 'Forward14', 'Fixed14', 'Reweight14'):
                path = folder / f'{name}.json'; evidence_paths.append(path)
                record = json.loads(path.read_text())
                pred_path = folder / f'{name}_predictions.npy'; evidence_paths.append(pred_path)
                accuracy = float(100 * np.mean(np.load(pred_path) == labels))
                np.testing.assert_allclose(accuracy, record['accuracy'], atol=1e-10, rtol=0)
                values[dataset][name].append(accuracy)
                if name == 'Forward14':
                    if len(record['history']) != 14:raise ValueError('Missing Forward learning-curve epochs')
                    histories[dataset].append(record['history'])
                if name == 'Fixed14':
                    assert record['same_pre_checkpoint'] and record['same_minibatch_and_dropout_rng']
                if name == 'Reweight14':
                    assert record['replay_predictions_exact']
                    np.testing.assert_array_equal(np.load(pred_path), np.load(reference / f'{mode}_Reweight-R_predictions.npy'))
                if name == 'CE4':
                    assert record['estimator_replay_exact']
                    diagnostic = record['diagnostics']; calibration[dataset].append(diagnostic)
                    fit = set(diagnostic['temperature_fit_validation_indices'])
                    check = set(diagnostic['calibration_check_validation_indices'])
                    assert not fit.intersection(check)
                    for item in diagnostic['anchor_sensitivity']:
                        matrices_by_variant[str(item['k'])].append(np.asarray(item['matrix']))
                    matrices_by_variant['calibrated10'].append(np.asarray(diagnostic['temperature_scaled_k10_matrix']))
                    k10 = next(item['matrix'] for item in diagnostic['anchor_sensitivity'] if item['k'] == 10)
                    np.testing.assert_array_equal(k10, np.load(reference / 'T_estimated.npy'))
            pred_path = reference / f'{mode}_Forward_predictions.npy'; evidence_paths.append(pred_path)
            values[dataset]['Forward4'].append(float(100 * np.mean(np.load(pred_path) == labels)))
            fixed_record = json.loads((folder / 'Fixed14.json').read_text())
            revised_record = json.loads((folder / 'Reweight14.json').read_text())
            revised_history = [row for row in revised_record['history'] if row['phase'] == 'revision']
            revision_diagnostics[dataset].append({
                'seed': seed,
                'fixed_selected_epoch': fixed_record['selected_revision_epoch'],
                'revised_selected_epoch': int(np.argmin([row['validation_nll'] for row in revised_history])),
                'prediction_disagreements': int(np.count_nonzero(
                    np.load(folder / 'Fixed14_predictions.npy') != np.load(folder / 'Reweight14_predictions.npy')))})
        for comparator, kind in [('Forward14', 'epoch-matched'), ('Fixed14', 'same-checkpoint')]:
            contrasts.append({'dataset': dataset, 'contrast': kind,
                              **paired_statistics(values[dataset][comparator], values[dataset]['Reweight14'])})
        if dataset in KNOWN_T:
            for variant, matrices in matrices_by_variant.items():
                errors = [matrix_errors(matrix, KNOWN_T[dataset]) for matrix in matrices]
                sensitivity.append({'dataset': dataset, 'variant': variant,
                    **{key + '_mean': float(np.mean([item[key] for item in errors])) for key in errors[0]},
                    **{key + '_std': float(np.std([item[key] for item in errors], ddof=0)) for key in errors[0]},
                    'uniform_baseline_frobenius': float(np.linalg.norm(KNOWN_T[dataset] - np.ones((3, 3)) / 3)),
                    'frobenius_relative_to_uniform': float(np.mean([item['frobenius'] for item in errors]) /
                                                           np.linalg.norm(KNOWN_T[dataset] - np.ones((3, 3)) / 3))})
    # A zero estimated paired variance does not justify a finite-t interval or
    # an equivalence claim. Keep such comparisons descriptive and non-rejecting
    # within the six-hypothesis adjustment family.
    for row in contrasts:
        row['inference_degenerate'] = row['difference_std_sample'] == 0
        if row['inference_degenerate']:
            row['p_value'] = row['ci95_low_pp'] = row['ci95_high_pp'] = None
    adjusted_values = holm_adjust([1.0 if row['inference_degenerate'] else row['p_value'] for row in contrasts])
    for row, adjusted in zip(contrasts, adjusted_values):
        row['p_value_holm'] = None if row['inference_degenerate'] else adjusted
    cifar_controls = []
    with np.load(data_dir / 'CIFAR.npz') as source:labels = source['Yts']
    for seed in range(runs):
        folder = controls / 'cifar_controls' / f'seed_{seed:02d}'
        record = json.loads((folder / 'complete.json').read_text()); evidence_paths.append(folder / 'complete.json')
        for name in ('classifier_randomness', 'estimator_randomness'):
            path = folder / f'{name}_predictions.npy'; evidence_paths.append(path)
            np.testing.assert_allclose(100 * np.mean(np.load(path) == labels), record[name + '_accuracy'], atol=1e-10, rtol=0)
        cifar_controls.append(record)
    summary = {'runs': runs, 'accuracies': values, 'contrasts': contrasts, 'sensitivity': sensitivity,
               'calibration': calibration, 'cifar_controls': cifar_controls,
               'saved_diagnostics': saved, 'forward_histories': histories,
               'revision_diagnostics': revision_diagnostics,
               'learning_curve_summary': {
                   dataset: {
                       'selected_epochs': [int(np.argmin([r['validation_nll'] for r in history])) + 1
                                           for history in histories[dataset]],
                       'validation_nll_epoch4_mean': float(np.mean([h[3]['validation_nll'] for h in histories[dataset]])),
                       'validation_nll_selected_mean': float(np.mean([min(r['validation_nll'] for r in h)
                                                                      for h in histories[dataset]]))}
                   for dataset in DATASETS},
               'study_runtime': json.loads((controls / 'study_runtime.json').read_text())}
    (output / 'control_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(output / 'controlled_contrasts.csv', contrasts)
    write_csv(output / 'anchor_sensitivity.csv', sensitivity)
    accuracy_rows = []
    for dataset in DATASETS:
        for name, numbers in values[dataset].items():
            accuracy_rows.append({'dataset': dataset, 'method': name, 'n_runs': runs,
                                 'mean': float(np.mean(numbers)), 'std': float(np.std(numbers))})
    write_csv(output / 'controlled_accuracy.csv', accuracy_rows)
    tabular(output / 'accuracy.tex', ['Dataset', 'CE4', 'F4', 'F14', 'Fixed14', 'R14'],
            [[SHORT[d]] + [pm(values[d][name]) for name in ('CE4', 'Forward4', 'Forward14', 'Fixed14', 'Reweight14')]
             for d in DATASETS])
    tabular(output / 'contrasts.tex', ['Dataset', 'Control', '$\\Delta$ (pp)', '95\\% CI', 'Holm $p$'],
            [[SHORT[row['dataset']], 'F14' if row['contrast'] == 'epoch-matched' else 'Fixed14',
              f"${row['mean_difference_pp']:+.3f}$",
              '---' if row['inference_degenerate'] else f"$[{row['ci95_low_pp']:.3f},{row['ci95_high_pp']:.3f}]$",
              '---' if row['inference_degenerate'] else
              ('$<0.00001$' if row['p_value_holm'] < 1e-5 else f"${row['p_value_holm']:.5f}$")]
             for row in contrasts])
    tabular(output / 'anchors.tex', ['Dataset', '$k=1$', '$k=5$', '$k=10$', '$k=20$', 'Scaled'],
            [[SHORT[d]] + [f"${next(r for r in sensitivity if r['dataset']==d and r['variant']==variant)['frobenius_mean']:.3f}"
                           f"\\pm{next(r for r in sensitivity if r['dataset']==d and r['variant']==variant)['frobenius_std']:.3f}$"
                          for variant in ('1', '5', '10', '20', 'calibrated10')] for d in DATASETS[:2]])
    cal_rows = []
    for d in DATASETS:
        diagnostics = calibration[d]
        cal_rows.append([SHORT[d], pm([r['temperature'] for r in diagnostics]),
            pm([r['uncalibrated']['nll'] for r in diagnostics], 3),
            pm([r['temperature_scaled']['nll'] for r in diagnostics], 3),
            pm([100 * r['uncalibrated']['ece'] for r in diagnostics]),
            pm([100 * r['temperature_scaled']['ece'] for r in diagnostics])])
    tabular(output / 'calibration.tex', ['Dataset', '$\\tau$', 'NLL raw', 'NLL scaled', 'ECE raw', 'ECE scaled'], cal_rows)
    macros = []
    for d, prefix in zip(DATASETS, ('FMa', 'FMb', 'CIF')):
        if d in KNOWN_T:
            row = next(r for r in sensitivity if r['dataset'] == d and r['variant'] == '10')
            macros.append('\\newcommand{\\' + prefix + 'UniformError}{' + f"{row['uniform_baseline_frobenius']:.4f}" + '}')
            macros.append('\\newcommand{\\' + prefix + 'UniformRelative}{' + f"{row['frobenius_relative_to_uniform']:.3f}" + '}')
        for kind, suffix in [('epoch-matched', 'Budget'), ('same-checkpoint', 'Revision')]:
            row = next(r for r in contrasts if r['dataset'] == d and r['contrast'] == kind)
            for key, name, dec in [('mean_difference_pp', 'Diff', 2), ('p_value_holm', 'P', 5),
                                   ('ci95_low_pp', 'Low', 2), ('ci95_high_pp', 'High', 2)]:
                value = '---' if row[key] is None else f'{row[key]:.{dec}f}'
                macros.append('\\newcommand{\\' + prefix + suffix + name + '}{' + value + '}')
    for name, field in [('Classifier', 'classifier_randomness_accuracy'), ('Estimator', 'estimator_randomness_accuracy')]:
        numbers = [r[field] for r in cifar_controls]
        macros.append('\\newcommand{\\Cifar' + name + 'Mean}{' + f'{np.mean(numbers):.2f}' + '}')
        macros.append('\\newcommand{\\Cifar' + name + 'SD}{' + f'{np.std(numbers):.2f}' + '}')
    seconds = summary['study_runtime']['wall_seconds']
    macros.append('\\newcommand{\\SupplementRuntime}{' + f'{seconds/60:.2f} minutes' + '}')
    (output / 'results_macros.tex').write_text('\n'.join(macros) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'aml2-control-mpl'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'aml2-control-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with plt.rc_context({'font.size': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300}):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.7), layout='constrained')
        for ax, d in zip(axes, DATASETS):
            for key, label, color, style in [('training_loss', 'Train', '#a04c31', '--'), ('validation_nll', 'Validation', '#2463a0', '-')]:
                array = np.array([[epoch[key] for epoch in history] for history in histories[d]])
                mean, sd = array.mean(0), array.std(0)
                ax.plot(range(1,15), mean, color=color, linestyle=style, label=label)
                ax.fill_between(range(1,15), mean-sd, mean+sd, color=color, alpha=.12)
            ax.axvline(4, color='.5', linewidth=.7, linestyle=':')
            ax.set(title=SHORT[d], xlabel='Epoch', xticks=[1,4,8,14])
        axes[0].set_ylabel('Forward NLL'); axes[0].legend(frameon=False, fontsize=9)
        fig.savefig(assets / 'forward_learning_curves.png'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.5, 3.2), layout='constrained')
        series = [('Original: split, T and classifier vary', values['CIFAR']['Forward4'], '#2463a0', 'o'),
                  ('Fixed split/T: classifier varies', [r['classifier_randomness_accuracy'] for r in cifar_controls], '#b65c16', 's'),
                  ('Fixed split/classifier: estimator varies', [r['estimator_randomness_accuracy'] for r in cifar_controls], '#378455', '^')]
        for label, numbers, color, marker in series:
            ax.plot(range(runs), numbers, marker=marker, color=color, linewidth=.8, markersize=3,
                    label=label + f' ({np.mean(numbers):.2f} ± {np.std(numbers):.2f})')
        ax.set(xlabel='Seed index', ylabel='Clean-test accuracy (%)', xticks=range(runs))
        ax.legend(frameon=False, fontsize=9, loc='lower left',
                  bbox_to_anchor=(0, 1.02), borderaxespad=0)
        fig.savefig(assets / 'cifar_control_variability.png'); plt.close(fig)
    provenance = {'baseline': str(baseline.resolve()), 'controls': str(controls.resolve()),
        'summary_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_paths},
        'figures': {'Figure 1': 'forward_learning_curves.png', 'Figure 2': 'cifar_control_variability.png'}}
    (assets / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (assets / 'README.txt').write_text('Assets for the feedback-updated report.\nFigure 1: forward_learning_curves.png\n'
        'Figure 2: cifar_control_variability.png\nGenerated only from the completed ten-seed controls and saved baseline.\n'
        'Copy both PNGs to that report project\'s images/ folder. Older paired/matrix figures belong to the earlier report.\n')
    print('Verified and summarized full ten-seed controls:', output)
    for row in contrasts:
        print(row['dataset'], row['contrast'], row['mean_difference_pp'], row['p_value_holm'])
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.baseline, args.controls, args.data_dir, args.out, args.assets)
