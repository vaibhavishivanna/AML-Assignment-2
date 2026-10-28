import numpy as np

from benchmark_execution import compare_outputs


def test_benchmark_comparison_detects_prediction_changes_even_with_same_counts(tmp_path):
    result, reference = tmp_path / "result", tmp_path / "reference"
    first = result / "CIFAR/seed_00"
    second = reference / "CIFAR/seed_00"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    name = "estimated_Forward_predictions.npy"
    np.save(first / name, [0, 1, 2])
    np.save(second / name, [0, 1, 2])
    assert compare_outputs(result, reference)["all_exact"]
    np.save(second / name, [1, 0, 2])
    comparison = compare_outputs(result, reference)
    assert not comparison["all_exact"]
    assert comparison["details"][0]["different_entries"] == 2


def test_benchmark_comparison_requires_reference_arrays(tmp_path):
    result, reference = tmp_path / "result", tmp_path / "reference"
    folder = result / "CIFAR/seed_00"
    folder.mkdir(parents=True)
    np.save(folder / "T_estimated.npy", np.eye(3))
    comparison = compare_outputs(result, reference)
    assert not comparison["all_exact"]
    assert comparison["details"][0]["status"] == "missing_reference"


def test_benchmark_comparison_detects_changed_normalization(tmp_path):
    result, reference = tmp_path / "result", tmp_path / "reference"
    first, second = result / "CIFAR/seed_00", reference / "CIFAR/seed_00"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    np.savez(first / "normalization.npz", mean=[0.5], std=[0.2])
    np.savez(second / "normalization.npz", mean=[0.5], std=[0.2])
    assert compare_outputs(result, reference)["all_exact"]
    np.savez(second / "normalization.npz", mean=[0.6], std=[0.2])
    assert not compare_outputs(result, reference)["all_exact"]
