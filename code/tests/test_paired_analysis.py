import csv
import json

import numpy as np
import pytest

from compare_runs import compare_runs
from paired_analysis import analyze_saved_runs, holm_adjust, paired_statistics


def write_pair(root, dataset, seed, forward, reweight):
    mode = "estimated" if dataset == "CIFAR" else "known"
    folder = root / dataset / f"seed_{seed:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    for method, accuracy in [("Forward", forward), ("Reweight-R", reweight)]:
        record = {"dataset": dataset, "seed": seed, "mode": mode,
                  "method": method, "accuracy": accuracy}
        (folder / f"{mode}_{method}.json").write_text(json.dumps(record))


def test_paired_inference_matches_analytic_three_pair_example():
    # Differences 1,2,3 have mean 2, sample SD 1, t=sqrt(12), df=2.
    # The exact two-sided p-value is 1-sqrt(6/7).
    row = paired_statistics([40, 45, 50], [41, 47, 53])
    assert row["mean_difference_pp"] == 2
    assert row["difference_std_sample"] == 1
    assert row["p_value"] == pytest.approx(1 - np.sqrt(6 / 7), abs=1e-12)
    half = 4.302652729911275 / np.sqrt(3)
    assert row["ci95_low_pp"] == pytest.approx(2 - half, abs=1e-9)
    assert row["ci95_high_pp"] == pytest.approx(2 + half, abs=1e-9)


def test_holm_is_monotone_and_preserves_input_order():
    assert holm_adjust([.03, .001, .02]) == pytest.approx([.04, .003, .04])


@pytest.mark.parametrize("difference,expected_p", [(0, 1), (1, 0)])
def test_constant_differences_export_finite_values(difference, expected_p):
    row = paired_statistics([50, 50, 50], [50 + difference] * 3)
    assert row["p_value"] == expected_p
    assert row["ci95_low_pp"] == row["ci95_high_pp"] == difference
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("forward,reweight", [([1], [2]), ([1, 2], [3]),
                                                       ([np.nan, 2], [3, 4]), ([1, 2], [3, 101])])
def test_invalid_pairs_are_rejected(forward, reweight):
    with pytest.raises(ValueError):
        paired_statistics(forward, reweight)


def test_pairs_use_actual_nonzero_seed_and_not_file_order(tmp_path):
    write_pair(tmp_path, "CIFAR", 12, 50, 54)
    write_pair(tmp_path, "CIFAR", 7, 60, 58)
    row = analyze_saved_runs(tmp_path)[0]
    assert row["seeds"] == [7, 12]
    assert row["differences_pp"] == [-2, 4]
    assert row["mean_difference_pp"] == 1


def test_missing_pair_is_an_error_instead_of_silently_dropping_a_seed(tmp_path):
    write_pair(tmp_path, "CIFAR", 7, 60, 61)
    write_pair(tmp_path, "CIFAR", 8, 60, 62)
    (tmp_path / "CIFAR/seed_08/estimated_Reweight-R.json").unlink()
    with pytest.raises(ValueError, match="missing paired"):
        analyze_saved_runs(tmp_path)


def test_analysis_exports_all_three_comparisons_and_per_seed_evidence(tmp_path):
    results, output = tmp_path / "experiment", tmp_path / "analysis"
    for dataset in ["FashionMNIST0.3", "FashionMNIST0.6", "CIFAR"]:
        write_pair(results, dataset, 7, 60, 61)
        write_pair(results, dataset, 8, 60, 63)
    compare_runs(results, output)
    rows = list(csv.DictReader((output / "paired_comparison.csv").open()))
    assert len(rows) == 3 and all("p_value_holm" in row for row in rows)
    assert len(list(csv.DictReader((output / "paired_per_run.csv").open()))) == 6
    assert len(json.loads((output / "paired_analysis.json").read_text())["comparisons"]) == 3
    assert "Holm" in (output / "paired_comparison.tex").read_text()
