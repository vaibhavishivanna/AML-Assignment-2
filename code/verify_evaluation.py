"""Independently audit one completed evaluation, including nonzero seed starts.

    python verify_evaluation.py --results reproduced_results
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from transition_matrix import KNOWN_T, matrix_errors, validate_transition
from experiments.data_utils import default_data_dir

EXPECTED = {"FashionMNIST0.3": 18000, "FashionMNIST0.6": 18000, "CIFAR": 15000}
METHODS = {"Forward", "Reweight-R"}


def verify_classifier_matrices(folder, dataset, mode, method, estimate, record):
    """Check the actual saved classifier inputs, including their orientation."""
    prefix = folder / f"{mode}_{method}"
    matrix = validate_transition(np.load(str(prefix) + "_T_input.npy"))
    expected = KNOWN_T[dataset] if mode == "known" else estimate
    np.testing.assert_allclose(
        matrix, expected, rtol=0, atol=1e-7,
        err_msg=f"{prefix}: input must equal the supplied/estimated clean-row matrix; do not transpose it")
    if method != "Reweight-R":
        return
    revised = validate_transition(np.load(str(prefix) + "_T_revised.npy"))
    np.testing.assert_allclose(
        record["matrix_change_frobenius"], np.linalg.norm(revised - matrix), rtol=0, atol=1e-8)
    if dataset in KNOWN_T:
        for key, value in [("input_matrix_errors", matrix), ("revised_matrix_errors", revised)]:
            expected_errors = matrix_errors(value, KNOWN_T[dataset])
            for metric, error in expected_errors.items():
                np.testing.assert_allclose(record[key][metric], error, rtol=0, atol=1e-8)


def seed_folders(root, dataset):
    """Discover actual seed numbers; do not assume the first seed is zero."""
    folders = []
    for path in (Path(root) / dataset).glob("seed_*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"seed_(\d+)", path.name)
        assert match, f"Invalid experiment folder: {path}"
        folders.append((int(match.group(1)), path))
    folders.sort()
    assert folders, f"No saved seed folders for {dataset}"
    assert len({seed for seed, _ in folders}) == len(folders), f"Duplicate seed folders for {dataset}"
    return folders


def verify(root, data_dir):
    root, data_dir = Path(root), Path(data_dir)
    summary_file = root / "accuracy_summary.csv"
    if not summary_file.exists():
        raise FileNotFoundError(f"Missing {summary_file}. Run summarize_evaluation.py first.")
    with summary_file.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows, "Accuracy summary is empty"
    datasets = sorted({row["dataset"] for row in rows})
    assert set(datasets).issubset(EXPECTED), f"Unexpected datasets: {set(datasets) - set(EXPECTED)}"
    totals = {key: 0 for key in ["splits", "matrices", "predictions", "inputs", "revised", "labels", "pre_revision"]}
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        modes = sorted({row["mode"] for row in dataset_rows})
        valid_modes = {"estimated"} if dataset == "CIFAR" else {"known", "estimated"}
        assert set(modes).issubset(valid_modes), f"{dataset}: invalid matrix modes {modes}"
        if dataset in KNOWN_T:
            assert "known" in modes, f"{dataset}: missing known-matrix evaluation"
        for mode in modes:
            mode_rows = [row for row in dataset_rows if row["mode"] == mode]
            assert len(mode_rows) == 2 and {row["method"] for row in mode_rows} == METHODS, (
                f"{dataset}/{mode}: expected one summary for each classifier")
        folders = seed_folders(root, dataset)
        assert all(int(row["n_runs"]) == len(folders) for row in dataset_rows), (
            f"{dataset}: summary run counts do not match saved seed folders")
        labels = None
        source = data_dir / f"{dataset}.npz"
        if source.is_file():
            with np.load(source) as data:
                labels = np.asarray(data["Yts"])
            assert labels.shape == (3000,) and np.all(np.isin(labels, [0, 1, 2])), (dataset, "invalid clean labels")
        else:
            print(f"NOTE: clean labels unavailable for {dataset}: {source}")
        values = {(mode, method): [] for mode in modes for method in sorted(METHODS)}
        for seed, folder in folders:
            n = EXPECTED[dataset]
            with np.load(folder / "split_indices.npz") as split:
                train, valid = split["train"], split["validation"]
            assert len(train) == int(.8 * n) and len(valid) == n - int(.8 * n), (dataset, seed)
            np.testing.assert_array_equal(np.sort(np.r_[train, valid]), np.arange(n))
            assert len(np.unique(train)) == len(train) and len(np.unique(valid)) == len(valid)
            totals["splits"] += 1
            estimate = validate_transition(np.load(folder / "T_estimated.npy"))
            estimator = json.loads((folder / "estimator.json").read_text())
            if dataset in KNOWN_T:
                for metric, error in matrix_errors(estimate, KNOWN_T[dataset]).items():
                    np.testing.assert_allclose(estimator["errors"][metric], error, rtol=0, atol=1e-8)
            totals["matrices"] += 1
            for mode in modes:
                for method in sorted(METHODS):
                    stem = folder / f"{mode}_{method}"
                    record = json.loads(stem.with_suffix(".json").read_text())
                    assert (record["dataset"], record["seed"], record["mode"], record["method"]) == (
                        dataset, seed, mode, method), f"{stem}: record identity does not match saved seed"
                    verify_classifier_matrices(folder, dataset, mode, method, estimate, record)
                    totals["inputs"] += 1
                    if method == "Reweight-R":
                        totals["revised"] += 1
                        pre = np.load(str(stem) + "_pre_revision_predictions.npy")
                        assert pre.shape == (3000,) and np.all(np.isin(pre, [0, 1, 2]))
                        np.testing.assert_allclose(record["revision_gain_pp"],
                            record["accuracy"] - record["accuracy_before_revision"], rtol=0, atol=1e-8)
                        if labels is not None:
                            np.testing.assert_allclose(record["accuracy_before_revision"],
                                100 * np.mean(pre == labels), rtol=0, atol=1e-8)
                            totals["pre_revision"] += 1
                    predictions = np.load(str(stem) + "_predictions.npy")
                    assert predictions.shape == (3000,) and np.all(np.isin(predictions, [0, 1, 2]))
                    assert record["n_test"] == len(predictions)
                    accuracy = float(record["accuracy"])
                    assert np.isfinite(accuracy) and 0 <= accuracy <= 100
                    totals["predictions"] += 1
                    values[(mode, method)].append(accuracy)
                    if labels is not None:
                        np.testing.assert_allclose(accuracy, 100 * np.mean(predictions == labels), rtol=0, atol=1e-8)
                        totals["labels"] += 1
        for row in dataset_rows:
            accuracies = np.asarray(values[(row["mode"], row["method"])])
            np.testing.assert_allclose(float(row["accuracy_mean"]), accuracies.mean(), rtol=0, atol=1e-10)
            np.testing.assert_allclose(float(row["accuracy_std"]), accuracies.std(ddof=0), rtol=0, atol=1e-10)
            print(f"{dataset:17s} {row['method']:10s} ({row['mode']}) "
                  f"{accuracies.mean():6.2f} +/- {accuracies.std(ddof=0):5.2f}% ({len(folders)} runs)")
    print(f"PASS: {totals['splits']} splits, {totals['matrices']} estimated matrices, "
          f"{totals['predictions']} predictions, {len(rows)} summaries.")
    print(f"PASS: {totals['inputs']} classifier input matrices match their clean-row sources; "
          f"{totals['revised']} revised matrices and their errors/changes checked.")
    if totals["labels"]:
        print(f"PASS: Recomputed {totals['labels']}/{totals['predictions']} final and "
              f"{totals['pre_revision']} pre-revision accuracies using clean labels.")
    else:
        print("LIMITATION: No original .npz datasets found; test accuracies could not be independently recomputed.")
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Completed experiment to audit")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="Default: data/ beside this Python script")
    args = parser.parse_args()
    verify(args.results, args.data_dir)
