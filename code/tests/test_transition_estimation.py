import json
import numpy as np
import pytest
import torch

from algorithm.transition_estimation import estimate_from_probabilities, matrix_errors, validate_transition
from algorithm.reweight_t_revision import ReweightTRevision
from algorithm.revision_evaluation import EvaluatedReweightTRevision
from experiments.data_utils import KNOWN_T
from run_evaluation import main, split_indices


@pytest.mark.parametrize("name", list(KNOWN_T))
def test_anchor_estimation_recovers_known_T_from_exact_noisy_posteriors(name):
    T = KNOWN_T[name]
    # Pure and mixed examples, without labels.
    clean_posteriors = np.vstack([np.eye(3), [.2, .5, .3], [.7, .2, .1]])
    estimated, anchors = estimate_from_probabilities(clean_posteriors @ T)
    np.testing.assert_allclose(estimated, T, atol=1e-7)
    np.testing.assert_array_equal(anchors[:, 0], [0, 1, 2])


def test_average_anchor_variant_and_invalid_inputs():
    p = np.array([[.7, .2, .1], [.6, .3, .1], [.1, .8, .1], [.1, .1, .8]])
    T, anchors = estimate_from_probabilities(p, top_k=2)
    np.testing.assert_allclose(T[0], [.65, .25, .1])
    assert anchors.shape == (3, 2)
    for bad in [np.empty((0, 3)), np.ones((3, 3)), np.array([[np.nan, 0, 1]])]:
        with pytest.raises(ValueError):
            estimate_from_probabilities(bad)
    with pytest.raises(ValueError):
        estimate_from_probabilities(p, top_k=5)
    with pytest.raises(ValueError):
        validate_transition(np.eye(2))
    assert matrix_errors(KNOWN_T['FashionMNIST0.3'], KNOWN_T['FashionMNIST0.3'])["frobenius"] == 0


def test_saved_split_policy_matches_original_and_is_disjoint():
    tr, va = split_indices(18000, 4)
    assert len(tr) == 14400 and len(va) == 3600
    assert not np.intersect1d(tr, va).size
    np.testing.assert_array_equal(np.sort(np.r_[tr, va]), np.arange(18000))
    assert not np.array_equal(tr, split_indices(18000, 5)[0])


def test_evaluation_hook_preserves_vaibhavi_classifier():
    rng = np.random.RandomState(5)
    X = rng.normal(size=(30, 1, 28, 28)).astype(np.float32)
    S = rng.randint(0, 3, 30)
    args = (X[:24], S[:24], KNOWN_T['FashionMNIST0.3'], X[24:], S[24:])
    original = ReweightTRevision(epochs=1, warm=1, rev_epochs=1).fit(*args, seed=3)
    audited = EvaluatedReweightTRevision(epochs=1, warm=1, rev_epochs=1).fit(*args, seed=3)
    np.testing.assert_array_equal(original.predict_proba(X), audited.predict_proba(X))
    np.testing.assert_array_equal(original.T_revised_, audited.T_revised_)
    assert audited.pre_revision_model_ is not audited.model


def test_cifar_pipeline_saves_matrices_predictions_and_resumes(tmp_path, monkeypatch):
    rng = np.random.RandomState(7)
    data, out = tmp_path / "data", tmp_path / "results"
    data.mkdir()
    np.savez(data / "CIFAR.npz", Xtr=rng.randint(0, 256, (30, 32, 32, 3), dtype=np.uint8),
             Str=np.arange(30) % 3, Xts=rng.randint(0, 256, (6, 32, 32, 3), dtype=np.uint8), Yts=np.arange(6) % 3)
    argv = ["--data-dir", str(data), "--out", str(out), "--datasets", "CIFAR", "--runs", "1",
            "--epochs", "1", "--estimator-epochs", "1", "--warm", "1", "--revision-epochs", "1", "--workers", "1"]
    # Reuse initial estimates during full evaluation.
    main(argv + ["--estimate-only"])
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_cleanup"] = {"note": "Comments shortened"}
    manifest_path.write_text(json.dumps(manifest))
    folder = out / "CIFAR/seed_00"
    assert (folder / "T_estimated.npy").exists() and not (folder / "metrics.json").exists()
    assert not (folder / "noisy_model.pt").exists()
    assert not (out / "accuracy_table.tex").exists()
    from algorithm.transition_estimation import AnchorTransitionEstimator
    monkeypatch.setattr(AnchorTransitionEstimator, "fit", lambda *a, **k: pytest.fail("Estimation should be cached"))
    main(argv)
    assert json.loads(manifest_path.read_text())["source_cleanup"] == manifest["source_cleanup"]
    for name in ["estimated_Forward_predictions.npy", "estimated_Reweight-R_T_revised.npy",
                 "estimated_Reweight-R_pre_revision_predictions.npy", "metrics.json"]:
        assert (folder / name).exists()
    for name in ["CIFAR_estimated_T_revised_mean.npy", "CIFAR_estimated_T_revised_std.npy", "revision_table.tex"]:
        assert (out / name).exists()
    import csv
    row = next(csv.DictReader((out / "revision_summary.csv").open()))
    assert sum(int(row[k]) for k in ["improved", "worsened", "unchanged"]) == 1
    main(argv)
    assert len(list(out.glob("CIFAR/seed_*/metrics.json"))) == 1
    runtime = json.loads((out / "runtime_measured.json").read_text())
    assert runtime["wall_seconds"] > 0
    assert runtime["completed_splits_already_present_at_start"] == 1
    with pytest.raises(SystemExit):
        main(argv + ["--epochs", "2"])


def test_parallel_estimation_retains_all_seeds(tmp_path):
    data, out = tmp_path / "data", tmp_path / "parallel"
    data.mkdir()
    rng = np.random.RandomState(9)
    np.savez(data / "CIFAR.npz", Xtr=rng.randint(0, 256, (30, 32, 32, 3), dtype=np.uint8),
             Str=np.arange(30) % 3, Xts=rng.randint(0, 256, (6, 32, 32, 3), dtype=np.uint8), Yts=np.arange(6) % 3)
    main(["--data-dir", str(data), "--out", str(out), "--datasets", "CIFAR", "--runs", "2",
          "--epochs", "1", "--estimator-epochs", "1", "--warm", "1", "--revision-epochs", "1",
          "--workers", "2", "--threads", "1", "--estimate-only"])
    paths = sorted(out.glob("CIFAR/seed_*/T_estimated.npy"))
    assert len(paths) == 2
    assert not np.array_equal(np.load(paths[0]), np.load(paths[1]))
