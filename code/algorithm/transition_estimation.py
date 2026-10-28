"""Anchor-based T estimation (Xia et al., 2019); T[i,j] = P(noisy=j | clean=i)."""
import numpy as np

from .ce_baseline import CrossEntropyBaseline
from transition_matrix import matrix_errors, validate_transition


def estimate_from_probabilities(probabilities, top_k=1):
    """Estimate each row from top-k noisy posteriors; k=1 uses one anchor."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or len(p) == 0:
        raise ValueError("Expected a nonempty (n_examples, n_classes) array")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Posteriors must be finite and nonnegative")
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Posterior rows must sum to one")
    if not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= len(p):
        raise ValueError("top_k must be an integer in [1, n_examples]")
    # Break ties reproducibly.
    anchors = np.stack([np.argsort(-p[:, i], kind="stable")[:top_k]
                        for i in range(p.shape[1])])
    T = np.stack([p[indices].mean(0) for indices in anchors])
    T /= T.sum(1, keepdims=True)
    validate_transition(T, p.shape[1])
    return T.astype(np.float32), anchors


class AnchorTransitionEstimator:
    """Fit a noisy-label CNN, then estimate T from training anchors."""
    def __init__(self, epochs=10, top_k=1, **classifier_kwargs):
        self.epochs, self.top_k = epochs, top_k
        self.classifier_kwargs = classifier_kwargs

    def fit(self, Xtr, Str, Xval, Sval, seed=0):
        self.model_ = CrossEntropyBaseline(epochs=self.epochs, **self.classifier_kwargs)
        # Identity T makes validation loss ordinary noisy-label CE.
        self.model_.fit(Xtr, Str, np.eye(3, dtype=np.float32), Xval, Sval, seed)
        self.T_, self.anchor_indices_ = estimate_from_probabilities(
            self.model_.predict_proba(Xtr), self.top_k)
        return self
