"""Save the pre-revision checkpoint for paired evaluation."""
import copy

from .reweight_t_revision import ReweightTRevision


class EvaluatedReweightTRevision(ReweightTRevision):
    def _fit(self, *args):
        self._stage_count = 0
        super()._fit(*args)

    def _train(self, model, *args, **kwargs):
        result = super()._train(model, *args, **kwargs)
        self._stage_count += 1
        if self._stage_count == 2:
            self.pre_revision_model_ = copy.deepcopy(model).eval()
        return result
