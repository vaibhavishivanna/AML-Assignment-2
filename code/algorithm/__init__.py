from .ce_baseline import CrossEntropyBaseline
from .forward_correction import ForwardCorrection
from .backward_correction import BackwardCorrection
from .reweight_t_revision import ReweightTRevision

METHODS = {"CE": CrossEntropyBaseline, "Forward": ForwardCorrection,
           "Backward": BackwardCorrection, "T-Revision": ReweightTRevision}
