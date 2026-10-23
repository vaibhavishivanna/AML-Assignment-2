from .base import BaseClassifier
from .losses import backward_loss


class BackwardCorrection(BaseClassifier):
    """Backward loss correction (Patrini et al., CVPR 2017): L = sum_j (T^-1)_{sj} * (-log p_j)."""
    name = "Backward"
    loss_fn = staticmethod(backward_loss)
