
"""Verify saved Assignment 2 evaluation results.

Examples:
    python verify_evaluation.py
    python verify_evaluation.py --results revision_forward_10seeds
    python verify_evaluation.py --results cifar_k10_evaluation
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from transition_matrix import KNOWN_T, matrix_errors, validate_transition
from experiments.data_utils import default_data_dir


EXPECTED = {
    "FashionMNIST0.3": 18000,
    "FashionMNIST0.6": 18000,
    "CIFAR": 15000,
}

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


def verify(root, data_dir):
    summary_file = root / "accuracy_summary.csv"

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Missing {summary_file}. Run summarize_evaluation.py first."
        )

    with summary_file.open(newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows, "Accuracy summary is empty"

    datasets = sorted({row["dataset"] for row in rows})

    assert set(datasets).issubset(EXPECTED), (
        f"Unexpected datasets: {set(datasets) - set(EXPECTED)}"
    )

    total_splits = 0
    total_matrices = 0
    total_predictions = 0
    checked_labels = 0
    total_inputs = total_revised = checked_pre_revision = 0

    for dataset in datasets:
        dataset_rows = [
            row for row in rows if row["dataset"] == dataset
        ]

        methods = {row["method"] for row in dataset_rows}

        assert methods == METHODS, (
            f"{dataset}: expected {METHODS}, got {methods}"
        )

        assert len(dataset_rows) == 2, (
            f"{dataset}: expected exactly 2 classifier summaries"
        )

        n_runs = int(dataset_rows[0]["n_runs"])

        assert n_runs >= 1, f"{dataset}: invalid number of runs"
        assert all(
            int(row["n_runs"]) == n_runs
            for row in dataset_rows
        ), f"{dataset}: inconsistent run counts"

        n = EXPECTED[dataset]

        source = data_dir / f"{dataset}.npz"
        labels = None

        if source.is_file():
            with np.load(source) as data:
                labels = np.asarray(data["Yts"])

            assert labels.shape == (3000,), (
                dataset, labels.shape
            )

            assert np.all(np.isin(labels, [0, 1, 2]))

        else:
            print(
                f"NOTE: clean labels unavailable for {dataset}: "
                f"{source}"
            )

        mode = "estimated" if dataset == "CIFAR" else "known"

        for seed in range(n_runs):
            folder = root / dataset / f"seed_{seed:02d}"

            assert folder.is_dir(), (
                f"Missing experiment folder: {folder}"
            )

            # Verify train/validation split
            with np.load(folder / "split_indices.npz") as split:
                train = split["train"]
                valid = split["validation"]

            expected_train = int(0.8 * n)

            assert len(train) == expected_train
            assert len(valid) == n - expected_train

            assert np.array_equal(
                np.sort(np.r_[train, valid]),
                np.arange(n)
            ), (dataset, seed)

            assert len(np.unique(train)) == len(train)
            assert len(np.unique(valid)) == len(valid)

            total_splits += 1

            # Verify estimated transition matrix
            T = validate_transition(np.load(folder / "T_estimated.npy"))
            estimator = json.loads((folder / "estimator.json").read_text())
            if dataset in KNOWN_T:
                for metric, error in matrix_errors(T, KNOWN_T[dataset]).items():
                    np.testing.assert_allclose(estimator["errors"][metric], error, rtol=0, atol=1e-8)

            total_matrices += 1

            # Verify both classifiers
            for method in sorted(METHODS):
                stem = folder / f"{mode}_{method}"

                record = json.loads(
                    stem.with_suffix(".json").read_text()
                )
                assert (record["dataset"], record["seed"], record["mode"], record["method"]) == (
                    dataset, seed, mode, method)
                verify_classifier_matrices(folder, dataset, mode, method, T, record)
                total_inputs += 1
                if method == "Reweight-R":
                    total_revised += 1
                    pre = np.load(str(stem) + "_pre_revision_predictions.npy")
                    assert pre.shape == (3000,) and np.all(np.isin(pre, [0, 1, 2]))
                    np.testing.assert_allclose(
                        record["revision_gain_pp"],
                        record["accuracy"] - record["accuracy_before_revision"], rtol=0, atol=1e-8)
                    if labels is not None:
                        np.testing.assert_allclose(
                            record["accuracy_before_revision"], 100 * np.mean(pre == labels), rtol=0, atol=1e-8)
                        checked_pre_revision += 1

                predictions = np.load(
                    str(stem) + "_predictions.npy"
                )

                assert predictions.shape == (3000,)
                assert np.all(np.isin(predictions, [0, 1, 2]))
                assert record["n_test"] == len(predictions)

                total_predictions += 1

                # Independently recompute test accuracy
                if labels is not None:
                    recalculated = 100 * np.mean(
                        predictions == labels
                    )

                    assert np.isclose(
                        recalculated,
                        record["accuracy"],
                        atol=1e-8
                    ), (
                        dataset,
                        seed,
                        method,
                        recalculated,
                        record["accuracy"]
                    )

                    checked_labels += 1

        # Verify mean and standard deviation
        for row in dataset_rows:
            method = row["method"]
            row_mode = row["mode"]

            assert row_mode == mode, (
                dataset, row_mode, mode
            )

            values = np.array([
                json.loads(
                    (
                        root / dataset /
                        f"seed_{seed:02d}" /
                        f"{mode}_{method}.json"
                    ).read_text()
                )["accuracy"]
                for seed in range(n_runs)
            ])

            assert np.isclose(
                float(row["accuracy_mean"]),
                values.mean(),
                atol=1e-10
            )

            assert np.isclose(
                float(row["accuracy_std"]),
                values.std(ddof=0),
                atol=1e-10
            )

            print(
                f"{dataset:17s} {method:10s} "
                f"{values.mean():6.2f} +/- "
                f"{values.std(ddof=0):5.2f}% "
                f"({n_runs} runs)"
            )

    # Final verification summary
    print(
        f"PASS: {total_splits} splits, "
        f"{total_matrices} estimated matrices, "
        f"{total_predictions} predictions, "
        f"{len(rows)} summaries."
    )
    print(f"PASS: {total_inputs} classifier input matrices match their clean-row sources; "
          f"{total_revised} revised matrices are row-stochastic and their errors/changes match.")

    if checked_labels:
        print(
            f"PASS: Recomputed "
            f"{checked_labels}/{total_predictions} "
            f"test accuracies using clean labels."
        )
        print(f"PASS: Recomputed {checked_pre_revision} pre-revision test accuracies and checked paired gains.")
    else:
        print(
            "LIMITATION: No original .npz datasets found. "
            "Test accuracy could not be independently recomputed."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    here = Path(__file__).resolve().parent

    parser.add_argument(
        "--results",
        type=Path,
        default=here / "evaluation_results"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Dataset directory; default: data/ beside this Python script"
    )

    args = parser.parse_args()

    verify(args.results, args.data_dir)
