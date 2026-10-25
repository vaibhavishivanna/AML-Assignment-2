import copy
import torch
import torch.nn as nn

from .base import BaseClassifier
from .losses import ce_loss, reweight_loss

EPS = 1e-7


class ReweightTRevision(BaseClassifier):
    """Importance reweighting + T-Revision (Xia et al., NeurIPS 2019), three stages:
      1. CE warm-up on noisy labels                            (`warm` epochs)
      2. importance-reweighting loss with fixed T               (`epochs` epochs)
      3. revision: learn Delta T (init 0) jointly with the classifier, loss uses T + Delta T
         (`rev_epochs` epochs). T + Delta T is clamped >= 0 and row-normalised.
    A revision epoch is accepted only if the noisy-validation forward NLL computed with the
    revised matrix is lower than the best so far (initially the value with the input T).
    """
    name = "T-Revision"

    def __init__(self, warm=3, rev_epochs=8, lr_rev=1e-4, lr_dT=1e-3, **kw):
        super().__init__(**kw)
        self.warm, self.rev_epochs, self.lr_rev, self.lr_dT = warm, rev_epochs, lr_rev, lr_dT
        self.T_revised_ = None

    @staticmethod
    def _apply(T, dT):
        Tr = torch.clamp(T + dT, min=0.0)
        return Tr / Tr.sum(1, keepdim=True).clamp_min(EPS)

    def _fit(self, Xt, St, Xv, Sv, T, rng):
        m = self.model
        self._train(m, Xt, St, Xv, Sv, T, ce_loss, T, self.warm, self.lr, rng)
        self._train(m, Xt, St, Xv, Sv, T, reweight_loss, T, self.epochs, self.lr, rng)
        self.T_revised_ = T.clone().numpy()
        dT = nn.Parameter(torch.zeros_like(T))
        best = self._val_fwd_nll(Xv, Sv, T)

        if self.verbose:
            print(f"Initial revision validation loss: {best:.6f}")
        best_state, best_dT = copy.deepcopy(m.state_dict()), dT.detach().clone()
        opt = torch.optim.Adam([{"params": m.parameters(), "lr": self.lr_rev, "weight_decay": self.wd},
                                {"params": [dT], "lr": self.lr_dT}])
        for ep in range(self.rev_epochs):
            m.train()
            for b in self._batches(len(Xt), rng):
                loss = reweight_loss(m(Xt[b]), St[b], self._apply(T, dT), detach_beta=False)
                opt.zero_grad(); loss.backward(); opt.step()
            v = self._val_fwd_nll(Xv, Sv, self._apply(T, dT).detach())
            if self.verbose:
                matrix_change = torch.norm(
                    self._apply(T, dT).detach() - T
                ).item()

                print(
                    f"[T-Rev] Epoch {ep + 1}: "
                    f"val_loss={v:.6f}, "
                    f"best_loss={best:.6f}, "
                    f"matrix_change={matrix_change:.8f}"
                )
            if v < best:
                best, best_state, best_dT = v, copy.deepcopy(m.state_dict()), dT.detach().clone()
        m.load_state_dict(best_state)
        self.T_revised_ = self._apply(T, best_dT).numpy()
