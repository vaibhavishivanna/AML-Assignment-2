import csv
import json
import shutil

import numpy as np
import pytest

import verify_evaluation
from transition_matrix import KNOWN_T, matrix_errors


@pytest.fixture
def nonzero_results(tmp_path, monkeypatch):
    """Complete saved evidence with seeds 7 and 8, without training a model."""
    root, data = tmp_path / "results", tmp_path / "data"
    data.mkdir()
    monkeypatch.setitem(verify_evaluation.EXPECTED, "CIFAR", 10)
    labels = np.arange(3000) % 3
    np.savez(data / "CIFAR.npz", Yts=labels)
    for seed in (7, 8):
        folder = root / "CIFAR" / f"seed_{seed:02d}"
        folder.mkdir(parents=True)
        indices = np.random.RandomState(seed).permutation(10)
        np.savez(folder / "split_indices.npz", train=indices[:8], validation=indices[8:])
        np.save(folder / "T_estimated.npy", np.eye(3))
        (folder / "estimator.json").write_text("{}")
        for method in ("Forward", "Reweight-R"):
            prefix = folder / f"estimated_{method}"
            np.save(str(prefix) + "_T_input.npy", np.eye(3))
            np.save(str(prefix) + "_predictions.npy", labels)
            record = {"dataset": "CIFAR", "seed": seed, "mode": "estimated", "method": method,
                      "accuracy": 100., "n_test": 3000}
            if method == "Reweight-R":
                np.save(str(prefix) + "_T_revised.npy", np.eye(3))
                np.save(str(prefix) + "_pre_revision_predictions.npy", labels)
                record.update(matrix_change_frobenius=0., accuracy_before_revision=100., revision_gain_pp=0.)
            prefix.with_suffix(".json").write_text(json.dumps(record))
    with (root / "accuracy_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "mode", "method", "n_runs", "accuracy_mean", "accuracy_std"])
        writer.writeheader()
        for method in ("Forward", "Reweight-R"):
            writer.writerow({"dataset": "CIFAR", "mode": "estimated", "method": method,
                             "n_runs": 2, "accuracy_mean": 100, "accuracy_std": 0})
    return root, data


def test_verifier_accepts_valid_nonzero_seed_start(nonzero_results):
    counts = verify_evaluation.verify(*nonzero_results)
    assert counts["splits"] == 2 and counts["predictions"] == counts["labels"] == 4
    assert counts["pre_revision"] == 2


def test_verifier_rejects_missing_seed_even_when_summaries_exist(nonzero_results):
    root, data = nonzero_results
    shutil.rmtree(root / "CIFAR/seed_08")
    with pytest.raises(AssertionError, match="run counts"):
        verify_evaluation.verify(root, data)


def test_verifier_rejects_record_with_wrong_seed_identity(nonzero_results):
    root, data = nonzero_results
    path = root / "CIFAR/seed_07/estimated_Forward.json"
    record = json.loads(path.read_text())
    record["seed"] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="identity"):
        verify_evaluation.verify(root, data)


def test_verifier_checks_both_fashion_matrix_modes(nonzero_results, monkeypatch):
    root, data = nonzero_results
    dataset = "FashionMNIST0.3"
    monkeypatch.setitem(verify_evaluation.EXPECTED, dataset, 10)
    (root / "CIFAR").rename(root / dataset)
    (data / "CIFAR.npz").rename(data / f"{dataset}.npz")
    matrix = KNOWN_T[dataset]
    for folder in (root / dataset).glob("seed_*"):
        np.save(folder / "T_estimated.npy", matrix)
        (folder / "estimator.json").write_text(json.dumps({"errors": matrix_errors(matrix, matrix)}))
        for method in ("Forward", "Reweight-R"):
            old = folder / f"estimated_{method}.json"
            record = json.loads(old.read_text())
            for mode in ("known", "estimated"):
                prefix = folder / f"{mode}_{method}"
                record.update(dataset=dataset, mode=mode)
                np.save(str(prefix) + "_T_input.npy", matrix)
                if mode == "known":
                    shutil.copyfile(folder / f"estimated_{method}_predictions.npy", str(prefix) + "_predictions.npy")
                if method == "Reweight-R":
                    np.save(str(prefix) + "_T_revised.npy", matrix)
                    record.update(input_matrix_errors=matrix_errors(matrix, matrix),
                                  revised_matrix_errors=matrix_errors(matrix, matrix))
                    if mode == "known":
                        shutil.copyfile(folder / f"estimated_{method}_pre_revision_predictions.npy",
                                        str(prefix) + "_pre_revision_predictions.npy")
                prefix.with_suffix(".json").write_text(json.dumps(record))
    summary = root / "accuracy_summary.csv"
    with summary.open() as stream:
        old_rows = list(csv.DictReader(stream))
    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(old_rows[0]))
        writer.writeheader()
        for mode in ("known", "estimated"):
            for row in old_rows:
                writer.writerow({**row, "dataset": dataset, "mode": mode})
    counts = verify_evaluation.verify(root, data)
    assert counts["splits"] == 2 and counts["predictions"] == counts["labels"] == 8
    assert counts["pre_revision"] == 4
