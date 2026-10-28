"""Shared transition-matrix contract, independent of the training libraries.

Assignment convention: T[i, j] = P(noisy=j | clean=i). Rows index clean
classes, columns index noisy classes, and each ROW sums to one. For row-vector
posteriors q_noisy = p_clean @ T. The lecture convention is T_lecture = T.T.
The supplied assignment matrices already use our convention: do not transpose
them before passing them to a classifier or comparing them with an estimate.
"""
import numpy as np


TRANSITION_CONVENTION = "T[clean, noisy]; q_noisy = p_clean @ T; row-stochastic"

KNOWN_T = {
    "FashionMNIST0.3": np.array([[0.7, 0.3, 0.0], [0.0, 0.7, 0.3], [0.3, 0.0, 0.7]]),
    "FashionMNIST0.6": np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]),
}


def validate_transition(T, n_classes=3):
    """Validate a clean-row/noisy-column matrix without changing its orientation.

    Stochasticity alone cannot detect a transpose of a doubly stochastic matrix
    (including both supplied matrices). Directional regression tests and known
    matrix comparisons are therefore also needed.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (n_classes, n_classes):
        raise ValueError(f"Expected a {n_classes}x{n_classes} transition matrix")
    if not np.isfinite(T).all() or (T < 0).any():
        raise ValueError("Transition entries must be finite and nonnegative")
    if not np.allclose(T.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Transition rows (clean classes) must sum to one")
    return T


def matrix_errors(estimate, truth):
    """Probability-scale errors; both matrices must use the same convention."""
    estimate, truth = validate_transition(estimate), validate_transition(truth)
    diff = estimate - truth
    return {"mae": float(np.abs(diff).mean()),
            "frobenius": float(np.linalg.norm(diff, ord="fro")),
            "relative_l1": float(np.abs(diff).sum() / np.abs(truth).sum())}
