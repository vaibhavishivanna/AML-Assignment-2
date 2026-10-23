from .base import BaseClassifier
from .losses import forward_loss


class ForwardCorrection(BaseClassifier):
    """Forward loss correction (Patrini et al., CVPR 2017): L = -log (T^T p)_s."""
    name = "Forward"
    loss_fn = staticmethod(forward_loss)
