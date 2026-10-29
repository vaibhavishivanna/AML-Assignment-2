from .ce_baseline import CrossEntropyBaseline
from .forward_correction import ForwardCorrection
from .backward_correction import BackwardCorrection
from .reweight_t_revision import ReweightTRevision

# Historical standalone key retained for compatibility; the classifier display
# label and canonical evaluation records use Reweight-R.
METHODS = {"CE": CrossEntropyBaseline, "Forward": ForwardCorrection,
           "Backward": BackwardCorrection, "T-Revision": ReweightTRevision}
