"""Estimate per-split matrices and evaluate Forward and Reweight-R."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from algorithm.forward_correction import ForwardCorrection
from algorithm.revision_evaluation import EvaluatedReweightTRevision
from algorithm.transition_estimation import AnchorTransitionEstimator, matrix_errors, validate_transition
from evaluation import top1_accuracy
from experiments.data_utils import KNOWN_T, load_dataset, standardize


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def split_indices(n, seed):
    indices = np.random.RandomState(seed).permutation(n)
    k = int(0.8 * n)
    return indices[:k], indices[k:]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_manifest(args):
    root = Path(__file__).resolve().parent
    sources = sorted((root / "algorithm").glob("*.py")) + [
        root / "experiments/data_utils.py", root / "evaluation.py", Path(__file__)]
    config = {k: getattr(args, k) for k in ["epochs", "estimator_epochs", "top_k", "warm",
              "revision_epochs", "fashion_estimated", "threads"]}
    return {"schema_version": 1, "config": config,
            "data_sha256": {d: file_hash(Path(args.data_dir) / (d + ".npz")) for d in args.datasets},
            "source_sha256": {str(p.relative_to(root)): file_hash(p) for p in sources},
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "std_ddof": 0, "seed_policy": "split seed = model seed = run number"}


def run_split(dataset, seed, config, data_dir, output, estimate_only=False):
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    folder = Path(output) / dataset / f"seed_{seed:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / "metrics.json"
    if completed.exists():
        return json.loads(completed.read_text())
    start = time.monotonic()
    X, S, Xtest, Ytest = load_dataset(Path(data_dir) / (dataset + ".npz"))
    train, val = split_indices(len(X), seed)
    np.savez_compressed(folder / "split_indices.npz", train=train, validation=val)
    np.savez(folder / "normalization.npz", mean=X[train].mean((0, 2, 3), keepdims=True),
             std=X[train].std((0, 2, 3), keepdims=True) + 1e-6)
    Xt, Xv, Xs = standardize(X[train], X[val], Xtest)
    St, Sv = S[train], S[val]
    # Estimate T without test data.
    initial_path = folder / "T_estimated.npy"
    estimator_path = folder / "estimator.json"
    if initial_path.exists() and estimator_path.exists():
        T_estimated = validate_transition(np.load(initial_path)).astype(np.float32)
        estimator_info = json.loads(estimator_path.read_text())
    else:
        est_start = time.monotonic()
        estimator = AnchorTransitionEstimator(epochs=config["estimator_epochs"], top_k=config["top_k"])
        estimator.fit(Xt, St, Xv, Sv, seed)
        T_estimated = estimator.T_
        np.save(initial_path, T_estimated)
        np.save(folder / "anchor_train_indices.npy", train[estimator.anchor_indices_])
        condition = float(np.linalg.cond(T_estimated))
        estimator_info = {"seconds": time.monotonic() - est_start,
                          "condition_number": condition if np.isfinite(condition) else None,
                          "row_argmax": T_estimated.argmax(1).tolist()}
        if dataset in KNOWN_T:
            estimator_info["errors"] = matrix_errors(T_estimated, KNOWN_T[dataset])
        atomic_json(estimator_path, estimator_info)
        del estimator
    if estimate_only:
        return {"dataset": dataset, "seed": seed, "estimator": estimator_info}
    # Use known T for FashionMNIST and per-split estimates for CIFAR.
    modes = {"known": KNOWN_T[dataset]} if dataset in KNOWN_T else {"estimated": T_estimated}
    if config["fashion_estimated"] and dataset in KNOWN_T:
        modes["estimated"] = T_estimated
    records = []
    for mode, T in modes.items():
        validate_transition(T)
        for method in ("Forward", "Reweight-R"):
            result_path = folder / f"{mode}_{method}.json"
            if result_path.exists():
                records.append(json.loads(result_path.read_text()))
                continue
            fit_start = time.monotonic()
            cls = ForwardCorrection if method == "Forward" else EvaluatedReweightTRevision
            kwargs = {"epochs": config["epochs"]}
            if method == "Reweight-R":
                kwargs.update(warm=config["warm"], rev_epochs=config["revision_epochs"])
            classifier = cls(**kwargs).fit(Xt, St, T, Xv, Sv, seed)
            # Evaluate on clean labels only after fitting.
            predictions = classifier.predict(Xs)
            prefix = folder / f"{mode}_{method}"
            np.save(str(prefix) + "_predictions.npy", predictions)
            record = {"dataset": dataset, "seed": seed, "mode": mode, "method": method,
                      "accuracy": top1_accuracy(predictions, Ytest), "n_test": len(Ytest),
                      "seconds": time.monotonic() - fit_start}
            np.save(str(prefix) + "_T_input.npy", T)
            if method == "Reweight-R":
                pre = classifier._logits(torch.as_tensor(Xs), classifier.pre_revision_model_).argmax(1).numpy()
                np.save(str(prefix) + "_pre_revision_predictions.npy", pre)
                record["accuracy_before_revision"] = top1_accuracy(pre, Ytest)
                record["revision_gain_pp"] = record["accuracy"] - record["accuracy_before_revision"]
                revised = validate_transition(classifier.T_revised_)
                np.save(str(prefix) + "_T_revised.npy", revised)
                record["matrix_change_frobenius"] = float(np.linalg.norm(revised - T))
                if dataset in KNOWN_T:
                    record["input_matrix_errors"] = matrix_errors(T, KNOWN_T[dataset])
                    record["revised_matrix_errors"] = matrix_errors(revised, KNOWN_T[dataset])
            atomic_json(result_path, record)
            records.append(record)
            print(f"{dataset} seed={seed} {mode} {method}: {record['accuracy']:.2f}% "
                  f"({time.monotonic() - fit_start:.0f}s)", flush=True)
            del classifier
    result = {"dataset": dataset, "seed": seed, "train_size": len(train), "validation_size": len(val),
              "estimator": estimator_info, "classifiers": records, "seconds": time.monotonic() - start}
    atomic_json(completed, result)
    return result


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--out", type=Path, default=root / "evaluation_results")
    parser.add_argument("--datasets", nargs="+", choices=[*KNOWN_T, "CIFAR"], default=[*KNOWN_T, "CIFAR"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--estimator-epochs", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warm", type=int, default=2)
    parser.add_argument("--revision-epochs", type=int, default=8)
    parser.add_argument("--fashion-estimated", action="store_true")
    parser.add_argument("--estimate-only", action="store_true", help="Stop after initial estimation; resume later without this flag")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.runs, args.epochs, args.estimator_epochs, args.top_k, args.warm,
           args.revision_epochs, args.workers, args.threads) < 1 or args.seed_start < 0:
        parser.error("Counts must be positive; seed-start must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        manifest = experiment_manifest(args)
    except FileNotFoundError as error:
        parser.error(f"Dataset missing: {error.filename}. Supply --data-dir PATH.")
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if "source_cleanup" in saved:
            manifest["source_cleanup"] = saved["source_cleanup"]
        if saved != manifest:
            parser.error("Output contains a different configuration, source, data, or environment. Choose a new --out.")
    atomic_json(manifest_path, manifest)
    jobs = [(d, s, manifest["config"], str(args.data_dir.resolve()), str(args.out.resolve()), args.estimate_only)
            for s in range(args.seed_start, args.seed_start + args.runs) for d in args.datasets]
    if args.workers == 1:
        for job in jobs:
            result = run_split(*job)
            print(f"Completed {result['dataset']} seed={result['seed']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_split, *job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                print(f"Completed {result['dataset']} seed={result['seed']}", flush=True)
    from summarize_evaluation import summarize_results
    summarize_results(args.out)


if __name__ == "__main__":
    main()
