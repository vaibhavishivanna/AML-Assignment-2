"""Optional controlled experiments for the report, separate from the timed pipeline.

python run_report_controls.py --baseline reproduced_results --out report_controls
Uses saved baseline splits/matrices and verifies estimator/Reweight-R replays.
Clean test labels are used only for final evaluation, never model selection.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp
import torch

from algorithm.base import BaseClassifier
from algorithm.ce_baseline import CrossEntropyBaseline
from algorithm.forward_correction import ForwardCorrection
from algorithm.revision_evaluation import EvaluatedReweightTRevision
from algorithm.reweight_t_revision import ReweightTRevision
from algorithm.transition_estimation import estimate_from_probabilities
from algorithm.losses import forward_loss
from experiments.data_utils import default_data_dir, load_dataset, standardize
from transition_matrix import KNOWN_T, matrix_errors

DATASETS = ('FashionMNIST0.3', 'FashionMNIST0.6', 'CIFAR')


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TraceMixin:
    def _train(self, model, Xt, St, Xv, Sv, T_sel, loss_fn, T_loss, epochs, lr, rng):
        self.phase = getattr(self, 'phase', 'fit')
        self.loss_sum, self.loss_count = 0., 0
        def logged_loss(logits, labels, matrix):
            loss = loss_fn(logits, labels, matrix)
            self.loss_sum += float(loss.detach()) * len(labels)
            self.loss_count += len(labels)
            return loss
        return super()._train(model, Xt, St, Xv, Sv, T_sel, logged_loss, T_loss, epochs, lr, rng)

    def _val_fwd_nll(self, Xv, Sv, T, model=None):
        value = super()._val_fwd_nll(Xv, Sv, T, model)
        if not hasattr(self, 'history'):
            self.history = []
        row = {'phase': getattr(self, 'phase', 'fit'), 'validation_nll': value}
        if getattr(self, 'loss_count', 0):
            row['training_loss'] = self.loss_sum / self.loss_count
        self.history.append(row)
        self.loss_sum, self.loss_count = 0., 0
        return value


class TracedCE(TraceMixin, CrossEntropyBaseline):
    pass


class TracedForward(TraceMixin, ForwardCorrection):
    pass


class TracedReweight(TraceMixin, EvaluatedReweightTRevision):
    def _train(self, model, Xt, St, Xv, Sv, T_sel, loss_fn, T_loss, epochs, lr, rng):
        self.phase = 'warm' if self._stage_count == 0 else 'fixed'
        value = super()._train(model, Xt, St, Xv, Sv, T_sel, loss_fn, T_loss, epochs, lr, rng)
        if self._stage_count == 2:
            self.pre_numpy_rng = rng.get_state()
            self.pre_torch_rng = torch.get_rng_state().clone()
            self.phase = 'revision'
        return value

    def _fit(self, Xt, St, Xv, Sv, T, rng):
        super()._fit(Xt, St, Xv, Sv, T, rng)
        self.end_numpy_rng = rng.get_state()
        self.end_torch_rng = torch.get_rng_state().clone()


def noisy_calibration_metrics(probabilities, labels, bins=15):
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    group = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = sum(np.mean(group == i) * abs(confidence[group == i].mean() - correct[group == i].mean())
              for i in range(bins) if np.any(group == i))
    return {'nll': float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-15)).mean()),
            'ece': float(ece), 'brier': float(np.square(probabilities - np.eye(3)[labels]).sum(1).mean())}


def estimator_diagnostics(logits_train, logits_valid, labels_valid, seed, dataset, raw_probabilities=None):
    order = np.random.RandomState(seed + 20000).permutation(len(labels_valid))
    fit, check = np.array_split(order, 2)
    def objective(log_temperature):
        z = logits_valid[fit] / np.exp(log_temperature)
        return float((logsumexp(z, axis=1) - z[np.arange(len(fit)), labels_valid[fit]]).mean())
    fitted = minimize_scalar(objective, bounds=(-np.log(10), np.log(10)), method='bounded')
    temperature = float(np.exp(fitted.x))
    probabilities = softmax(logits_train, axis=1) if raw_probabilities is None else raw_probabilities
    calibrated = softmax(logits_train / temperature, axis=1)
    rows = []
    for k in (1, 5, 10, 20):
        matrix, _ = estimate_from_probabilities(probabilities, k)
        row = {'k': k, 'matrix': matrix.tolist()}
        if dataset in KNOWN_T:
            row['errors'] = matrix_errors(matrix, KNOWN_T[dataset])
        rows.append(row)
    scaled_matrix, _ = estimate_from_probabilities(calibrated, 10)
    result = {'temperature': temperature, 'temperature_fit_success': bool(fitted.success),
              'temperature_fit_size': len(fit), 'calibration_check_size': len(check),
              'temperature_fit_validation_indices': fit.tolist(),
              'calibration_check_validation_indices': check.tolist(),
              'uncalibrated': noisy_calibration_metrics(softmax(logits_valid[check], axis=1), labels_valid[check]),
              'temperature_scaled': noisy_calibration_metrics(softmax(logits_valid[check] / temperature, axis=1), labels_valid[check]),
              'anchor_sensitivity': rows, 'temperature_scaled_k10_matrix': scaled_matrix.tolist()}
    if dataset in KNOWN_T:
        result['temperature_scaled_k10_errors'] = matrix_errors(scaled_matrix, KNOWN_T[dataset])
    return result


def fixed_continuation(pre_model, numpy_state, torch_state, Xt, St, Xv, Sv, T, epochs=8):
    """Freeze Delta T=0; otherwise match the revision optimizer, RNG and selection."""
    model = copy.deepcopy(pre_model)
    rng = np.random.RandomState()
    rng.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    Xt, St, Xv, Sv = map(torch.as_tensor, (Xt, St, Xv, Sv))
    matrix = torch.as_tensor(T, dtype=torch.float32)
    normalized = ReweightTRevision._apply(matrix, torch.zeros_like(matrix))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    helper = BaseClassifier()
    helper.model = model
    best = helper._val_fwd_nll(Xv, Sv, matrix)
    state, selected, history = copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for batch in helper._batches(len(Xt), rng):
            loss = forward_loss(model(Xt[batch]), St[batch], normalized)
            total += float(loss.detach()) * len(batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        value = helper._val_fwd_nll(Xv, Sv, normalized)
        history.append({'epoch': epoch, 'training_loss': total / len(Xt), 'validation_nll': value})
        if value < best:
            best, state, selected = value, copy.deepcopy(model.state_dict()), epoch
    end_torch = torch.get_rng_state().clone()
    end_numpy = rng.get_state()
    model.load_state_dict(state)
    return helper, history, selected, end_numpy, end_torch


def load_split(dataset, seed, baseline, data_dir):
    X, S, Xtest, Ytest = load_dataset(Path(data_dir) / f'{dataset}.npz')
    folder = Path(baseline) / dataset / f'seed_{seed:02d}'
    with np.load(folder / 'split_indices.npz') as split:
        train, valid = split['train'], split['validation']
    Xt, Xv, Xs = standardize(X[train], X[valid], Xtest)
    return Xt, S[train], Xv, S[valid], Xs, Ytest, folder


def score(classifier, Xs, Ytest, path):
    predictions = classifier.predict(Xs)
    np.save(path, predictions)
    return float(100 * np.mean(predictions == Ytest))


def main_case(dataset, seed, baseline, data_dir, output, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    folder = Path(output) / dataset / f'seed_{seed:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return json.loads((folder / 'complete.json').read_text())
    started = time.perf_counter()
    Xt, St, Xv, Sv, Xs, Ytest, source = load_split(dataset, seed, baseline, data_dir)
    mode = 'estimated' if dataset == 'CIFAR' else 'known'
    T = np.load(source / 'T_estimated.npy') if mode == 'estimated' else KNOWN_T[dataset]
    print(f'START {dataset} seed={seed}', flush=True)
    ce = TracedCE(epochs=4).fit(Xt, St, np.eye(3), Xv, Sv, seed)
    train_logits = ce._logits(torch.as_tensor(Xt)).numpy()
    valid_logits = ce._logits(torch.as_tensor(Xv)).numpy()
    # Use the exact float32 PyTorch softmax used by the baseline estimator.
    train_probabilities = ce.predict_proba(Xt)
    replay_matrix, _ = estimate_from_probabilities(train_probabilities, 10)
    np.testing.assert_array_equal(replay_matrix, np.load(source / 'T_estimated.npy'))
    np.save(folder / 'noisy_logits_train.npy', train_logits)
    np.save(folder / 'noisy_logits_valid.npy', valid_logits)
    ce_record = {'accuracy': score(ce, Xs, Ytest, folder / 'CE4_predictions.npy'), 'history': ce.history,
                 'diagnostics': estimator_diagnostics(train_logits.astype(float), valid_logits.astype(float), Sv,
                                                      seed, dataset, train_probabilities),
                 'estimator_replay_exact': True}
    save_json(folder / 'CE4.json', ce_record)
    print(f'{dataset} seed={seed}: CE4/estimator/calibration complete; acc={ce_record["accuracy"]:.2f}', flush=True)
    del ce
    forward = TracedForward(epochs=14).fit(Xt, St, T, Xv, Sv, seed)
    f_record = {'accuracy': score(forward, Xs, Ytest, folder / 'Forward14_predictions.npy'), 'history': forward.history}
    save_json(folder / 'Forward14.json', f_record)
    print(f'{dataset} seed={seed}: Forward14 complete; acc={f_record["accuracy"]:.2f}', flush=True)
    del forward
    reweight = TracedReweight(epochs=4, warm=2, rev_epochs=8).fit(Xt, St, T, Xv, Sv, seed)
    r_accuracy = score(reweight, Xs, Ytest, folder / 'Reweight14_predictions.npy')
    np.testing.assert_array_equal(np.load(folder / 'Reweight14_predictions.npy'),
                                  np.load(source / f'{mode}_Reweight-R_predictions.npy'))
    pre_predictions = reweight._logits(torch.as_tensor(Xs), reweight.pre_revision_model_).argmax(1).numpy()
    np.testing.assert_array_equal(pre_predictions, np.load(source / f'{mode}_Reweight-R_pre_revision_predictions.npy'))
    save_json(folder / 'Reweight14.json', {'accuracy': r_accuracy, 'history': reweight.history,
                                         'replay_predictions_exact': True})
    print(f'{dataset} seed={seed}: Reweight14 replay exact', flush=True)
    fixed, history, selected, end_numpy, end_torch = fixed_continuation(
        reweight.pre_revision_model_, reweight.pre_numpy_rng, reweight.pre_torch_rng,
        Xt, St, Xv, Sv, T)
    assert np.array_equal(end_numpy[1], reweight.end_numpy_rng[1]) and end_numpy[2:] == reweight.end_numpy_rng[2:]
    assert torch.equal(end_torch, reweight.end_torch_rng), 'Dropout RNG differs between revision branches'
    fixed_record = {'accuracy': score(fixed, Xs, Ytest, folder / 'Fixed14_predictions.npy'),
                    'history': history, 'selected_revision_epoch': selected,
                    'same_pre_checkpoint': True, 'same_minibatch_and_dropout_rng': True}
    save_json(folder / 'Fixed14.json', fixed_record)
    result = {'dataset': dataset, 'seed': seed, 'CE4': ce_record['accuracy'], 'Forward14': f_record['accuracy'],
              'Reweight14': r_accuracy, 'Fixed14': fixed_record['accuracy'], 'seconds': time.perf_counter() - started}
    save_json(folder / 'complete.json', result)
    print(f'DONE {dataset} seed={seed}: {result}', flush=True)
    return result


def cifar_case(seed, baseline, data_dir, output, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    folder = Path(output) / 'cifar_controls' / f'seed_{seed:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return json.loads((folder / 'complete.json').read_text())
    started = time.perf_counter()
    Xt, St, Xv, Sv, Xs, Ytest, source = load_split('CIFAR', 0, baseline, data_dir)
    T = np.load(source / 'T_estimated.npy')
    if seed == 0:
        reference = np.load(source / 'estimated_Forward_predictions.npy')
        np.save(folder / 'classifier_randomness_predictions.npy', reference)
        np.save(folder / 'estimator_randomness_predictions.npy', reference)
        accuracy_a = accuracy_b = float(100 * np.mean(reference == Ytest))
        matrix = T
    else:
        classifier = ForwardCorrection(epochs=4).fit(Xt, St, T, Xv, Sv, seed)
        accuracy_a = score(classifier, Xs, Ytest, folder / 'classifier_randomness_predictions.npy')
        del classifier
        preliminary = CrossEntropyBaseline(epochs=4).fit(Xt, St, np.eye(3), Xv, Sv, seed)
        matrix, _ = estimate_from_probabilities(preliminary.predict_proba(Xt), 10)
        del preliminary
        classifier = ForwardCorrection(epochs=4).fit(Xt, St, matrix, Xv, Sv, 0)
        accuracy_b = score(classifier, Xs, Ytest, folder / 'estimator_randomness_predictions.npy')
    np.save(folder / 'estimator_randomness_T.npy', matrix)
    result = {'seed': seed, 'split_seed': 0, 'classifier_randomness_accuracy': accuracy_a,
              'estimator_randomness_accuracy': accuracy_b, 'seconds': time.perf_counter() - started}
    save_json(folder / 'complete.json', result)
    print(f'DONE CIFAR controlled seed={seed}: {result}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=default_data_dir())
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'report_controls')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.runs, args.workers, args.threads) < 1:
        parser.error('Run/worker/thread counts must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    baseline_manifest = json.loads((args.baseline / 'manifest.json').read_text())
    expected = {'epochs': 4, 'estimator_epochs': 4, 'top_k': 10, 'warm': 2, 'revision_epochs': 8}
    if any(baseline_manifest['config'].get(key) != value for key, value in expected.items()):
        parser.error('Baseline does not match the report training settings')
    actual_data_hashes = {dataset: digest(args.data_dir / f'{dataset}.npz') for dataset in DATASETS}
    if any(actual_data_hashes[dataset] != baseline_manifest['data_sha256'][dataset] for dataset in DATASETS):
        parser.error('Datasets differ from the saved baseline')
    for dataset in DATASETS:
        for seed in range(args.runs):
            if not (args.baseline / dataset / f'seed_{seed:02d}' / 'metrics.json').exists():
                parser.error(f'Missing completed baseline: {dataset} seed={seed}')
    manifest = {'study_source_sha256': digest(__file__), 'baseline_manifest_sha256': digest(args.baseline / 'manifest.json'),
                'data_sha256': actual_data_hashes,
                'baseline': str(args.baseline.resolve()), 'runs': args.runs, 'threads': args.threads,
                'scope': 'Supplementary controls; canonical experiment unchanged',
                'source_sha256': {str(p.relative_to(Path(__file__).parent)): digest(p) for p in
                    sorted((Path(__file__).parent / 'algorithm').glob('*.py')) +
                    [Path(__file__).parent / 'experiments/data_utils.py', Path(__file__).parent / 'transition_matrix.py']}}
    manifest_path = args.out / 'study_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('Study output belongs to different source/settings; choose a fresh --out')
    save_json(manifest_path, manifest)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for seed in range(args.runs):
            for dataset in DATASETS:
                futures.append(pool.submit(main_case, dataset, seed, args.baseline, args.data_dir, args.out, args.threads))
            futures.append(pool.submit(cifar_case, seed, args.baseline, args.data_dir, args.out, args.threads))
        for future in as_completed(futures):
            future.result()
    save_json(args.out / 'study_runtime.json', {'wall_seconds': time.perf_counter() - started,
              'scope': 'Supplementary controls only, including any cached cases; excludes original 58-minute evaluation',
              'workers': args.workers, 'threads': args.threads})


if __name__ == '__main__':
    main()
