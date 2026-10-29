import numpy as np
import torch
import pytest

from algorithm.ce_baseline import CrossEntropyBaseline
from algorithm.revision_evaluation import EvaluatedReweightTRevision
from run_report_controls import TracedCE, TracedReweight, fixed_continuation, noisy_calibration_metrics
from transition_matrix import KNOWN_T
import algorithm.losses as losses


def test_tracing_preserves_ce_predictions():
    rng = np.random.RandomState(12)
    X = rng.randn(30, 1, 28, 28).astype(np.float32)
    labels = np.arange(30) % 3
    args = X[:24], labels[:24], np.eye(3), X[24:], labels[24:]
    original = CrossEntropyBaseline(epochs=2).fit(*args, seed=4)
    traced = TracedCE(epochs=2).fit(*args, seed=4)
    np.testing.assert_array_equal(original.predict_proba(X), traced.predict_proba(X))
    assert len(traced.history) == 2 and all('training_loss' in row for row in traced.history)


def test_revision_control_reuses_checkpoint_and_both_random_streams():
    rng = np.random.RandomState(13)
    X = rng.randn(30, 1, 28, 28).astype(np.float32)
    labels = np.arange(30) % 3
    matrix = KNOWN_T['FashionMNIST0.3']
    args = X[:24], labels[:24], matrix, X[24:], labels[24:]
    original = EvaluatedReweightTRevision(epochs=1, warm=1, rev_epochs=1).fit(*args, seed=4)
    traced = TracedReweight(epochs=1, warm=1, rev_epochs=1).fit(*args, seed=4)
    np.testing.assert_array_equal(original.predict_proba(X), traced.predict_proba(X))
    before = {name: value.clone() for name, value in traced.pre_revision_model_.state_dict().items()}
    control, history, selected, numpy_end, torch_end = fixed_continuation(
        traced.pre_revision_model_, traced.pre_numpy_rng, traced.pre_torch_rng,
        X[:24], labels[:24], X[24:], labels[24:], matrix, epochs=1)
    for name, value in before.items():
        torch.testing.assert_close(value, traced.pre_revision_model_.state_dict()[name], rtol=0, atol=0)
    np.testing.assert_array_equal(numpy_end[1], traced.end_numpy_rng[1])
    assert numpy_end[2:] == traced.end_numpy_rng[2:]
    assert torch.equal(torch_end, traced.end_torch_rng)
    assert len(history) == 1 and selected in (0, 1)
    assert control.predict(X).shape == (30,)


def test_calibration_scores_match_perfect_and_uniform_posteriors():
    labels = np.array([0, 1, 2])
    perfect = noisy_calibration_metrics(np.eye(3), labels)
    assert perfect == {'nll': 0., 'ece': 0., 'brier': 0.}
    uniform = noisy_calibration_metrics(np.full((3, 3), 1 / 3), labels)
    assert abs(uniform['nll'] - np.log(3)) < 1e-12
    assert abs(uniform['brier'] - 2 / 3) < 1e-12


@pytest.mark.parametrize('dataset', list(KNOWN_T))
def test_ideal_detached_population_gradient_matches_stationary_point_argument(dataset, monkeypatch):
    monkeypatch.setattr(losses, 'EPS', 0.)
    matrix = torch.tensor(KNOWN_T[dataset], dtype=torch.float64)
    truth = torch.tensor([.2, .3, .5], dtype=torch.float64)
    noisy = truth @ matrix
    for posterior in [truth, torch.tensor([.4, .2, .4], dtype=torch.float64)]:
        logits = posterior.log().reshape(1, 3).requires_grad_()
        risk = sum(noisy[s] * losses.reweight_loss(logits, torch.tensor([s]), matrix) for s in range(3))
        gradient, = torch.autograd.grad(risk, logits)
        implied = posterior @ matrix
        expected = posterior * ((noisy * posterior / implied).sum() - noisy / implied)
        torch.testing.assert_close(gradient[0], expected, atol=1e-14, rtol=0)
        if torch.equal(posterior, truth):
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-14, rtol=0)
        else:
            assert float(gradient.norm()) > 1e-3
