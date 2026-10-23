from .base import BaseClassifier
from .losses import ce_loss


class CrossEntropyBaseline(BaseClassifier):
    """No correction: plain CE on the noisy labels (reference point)."""
    name = "CE"
    loss_fn = staticmethod(ce_loss)
