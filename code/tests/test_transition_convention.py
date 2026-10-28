"""Directional checks: row sums alone cannot reveal a mistaken transpose."""
import numpy as np
import pytest
import torch

from algorithm.forward_correction import ForwardCorrection
from algorithm.losses import backward_loss, forward_loss, reweight_loss
from algorithm.reweight_t_revision import ReweightTRevision
from algorithm.transition_estimation import estimate_from_probabilities
from transition_matrix import KNOWN_T, matrix_errors, validate_transition
from verify_evaluation import verify_classifier_matrices


def test_fashion03_clean_zero_flips_to_noisy_one_not_two():
    T = torch.tensor(KNOWN_T["FashionMNIST0.3"], dtype=torch.float64)
    clean_zero = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    torch.testing.assert_close(clean_zero @ T, torch.tensor([[.7, .3, 0.]], dtype=torch.float64))
    p = torch.tensor([[.8, .15, .05]], dtype=torch.float64)
    s = torch.tensor([1])
    # P(S=1|x) = .3*.8 + .7*.15 + 0*.05 = .345.
    expected = -torch.log(torch.tensor(.345 + 1e-7, dtype=torch.float64))
    torch.testing.assert_close(forward_loss(p.log(), s, T), expected)
    assert not torch.isclose(forward_loss(p.log(), s, T.T), expected)


def test_losses_and_anchors_share_clean_row_convention():
    # Deliberately asymmetric and NOT doubly stochastic. Column normalization
    # or treating rows as noisy classes changes the resulting distributions.
    T = np.array([[.8, .15, .05], [.1, .7, .2], [.2, .1, .7]])
    p = torch.tensor([[.6, .3, .1]], dtype=torch.float64)
    matrix = torch.tensor(T, dtype=torch.float64)
    s = torch.tensor([1])
    q = p @ matrix
    torch.testing.assert_close(q.sum(1), torch.ones(1, dtype=torch.float64))
    expected_weight = p[0, 1] / (q[0, 1] + 1e-7)
    torch.testing.assert_close(reweight_loss(p.log(), s, matrix), expected_weight * -p[0, 1].log())
    for clean_class in range(3):
        expected_backward = sum(matrix[clean_class, noisy] * backward_loss(p.log(), torch.tensor([noisy]), matrix)
                                for noisy in range(3))
        torch.testing.assert_close(expected_backward, -p[0, clean_class].log())
    estimated, _ = estimate_from_probabilities(np.eye(3) @ T)
    np.testing.assert_allclose(estimated, T, atol=1e-7)
    with pytest.raises(ValueError, match="rows"):
        validate_transition(T.T)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.eye(2),
                                  np.full((3, 3), np.nan),
                                  np.array([[1.1, -.1, 0], [0, 1, 0], [0, 0, 1]])])
def test_classifier_rejects_invalid_transition_before_training(matrix):
    # No training is needed to reject an invalid transition matrix.
    with pytest.raises(ValueError):
        ForwardCorrection(epochs=1).fit(None, None, matrix, None, None)


def test_revision_keeps_rows_stochastic_when_clipped_or_tiny():
    T = torch.tensor(KNOWN_T["FashionMNIST0.3"], dtype=torch.float64)
    delta = torch.zeros_like(T)
    delta[0] = -T[0] - 1  # Completely clipped row must fall back to input.
    delta[1] = -T[1]
    delta[1, 0] = 1e-12   # Tiny positive row must still sum to one.
    delta.requires_grad_()
    revised = ReweightTRevision._apply(T, delta)
    torch.testing.assert_close(revised[0], T[0])
    torch.testing.assert_close(revised[1], torch.tensor([1., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(revised.sum(1), torch.ones(3, dtype=torch.float64))
    assert (revised >= 0).all()
    revised.square().sum().backward()
    assert torch.isfinite(delta.grad).all()


@pytest.mark.parametrize("mode,dataset", [("known", "FashionMNIST0.3"), ("estimated", "CIFAR")])
def test_saved_input_audit_detects_transpose_even_when_row_sums_pass(tmp_path, mode, dataset):
    T = KNOWN_T["FashionMNIST0.3"]
    prefix = tmp_path / f"{mode}_Forward_T_input.npy"
    np.save(prefix, T)
    verify_classifier_matrices(tmp_path, dataset, mode, "Forward", T, {})
    np.save(prefix, T.T)
    validate_transition(T.T)  # A transpose is also stochastic in this case.
    with pytest.raises(AssertionError, match="do not transpose"):
        verify_classifier_matrices(tmp_path, dataset, mode, "Forward", T, {})


def test_saved_revision_audit_checks_matrix_and_error_records(tmp_path):
    T = KNOWN_T["FashionMNIST0.3"]
    prefix = tmp_path / "known_Reweight-R"
    np.save(str(prefix) + "_T_input.npy", T)
    np.save(str(prefix) + "_T_revised.npy", T)
    record = {"matrix_change_frobenius": 0., "input_matrix_errors": matrix_errors(T, T),
              "revised_matrix_errors": matrix_errors(T, T)}
    verify_classifier_matrices(tmp_path, "FashionMNIST0.3", "known", "Reweight-R", T, record)
    record["matrix_change_frobenius"] = .1
    with pytest.raises(AssertionError):
        verify_classifier_matrices(tmp_path, "FashionMNIST0.3", "known", "Reweight-R", T, record)
    record["matrix_change_frobenius"] = 0.
    np.save(str(prefix) + "_T_revised.npy", np.zeros((3, 3)))
    with pytest.raises(ValueError):
        verify_classifier_matrices(tmp_path, "FashionMNIST0.3", "known", "Reweight-R", T, record)
