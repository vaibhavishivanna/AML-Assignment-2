"""Shared training loop. Subclasses only choose the loss (and, for T-Revision, extra stages).

Interface (identical for every classifier):
    clf.fit(Xtr, Str, T, Xval, Sval, seed)   X: float32 (N, C, H, W), already preprocessed
    clf.predict(X) -> labels
Model selection uses ONLY the noisy validation split: forward-corrected NLL under the supplied T.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .losses import ce_loss, forward_loss
from .networks import SmallCNN
from transition_matrix import validate_transition

torch.set_num_threads(2)


class BaseClassifier:
    name = "base"
    loss_fn = staticmethod(ce_loss)

    def __init__(self, epochs=10, lr=1e-3, wd=1e-4, batch=128, verbose=False):
        self.epochs, self.lr, self.wd, self.batch, self.verbose = epochs, lr, wd, batch, verbose
        self.model = None

    def _batches(self, n, rng):
        idx = rng.permutation(n)
        for i in range(0, n, self.batch):
            yield idx[i:i + self.batch]

    @torch.no_grad()
    def _logits(self, X, model=None):
        model = model or self.model
        model.eval()
        return torch.cat([model(X[i:i + 1024]) for i in range(0, len(X), 1024)])

    @torch.no_grad()
    def _val_fwd_nll(self, Xv, Sv, T, model=None):
        return forward_loss(self._logits(Xv, model), Sv, T).item()

    def _train(self, model, Xt, St, Xv, Sv, T_sel, loss_fn, T_loss, epochs, lr, rng):
        """Train `model`; restore the epoch with the lowest noisy-validation forward NLL under T_sel."""
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, best_state = float("inf"), copy.deepcopy(model.state_dict())
        for ep in range(epochs):
            model.train()
            for b in self._batches(len(Xt), rng):
                loss = loss_fn(model(Xt[b]), St[b], T_loss)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            v = self._val_fwd_nll(Xv, Sv, T_sel, model)
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"  [{self.name}] ep{ep + 1:02d} val_fwd_nll={v:.4f}")
        model.load_state_dict(best_state)
        return best

    def fit(self, Xtr, Str, T, Xval, Sval, seed=0):
        T = validate_transition(T)
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        Xt, Xv = torch.as_tensor(Xtr), torch.as_tensor(Xval)
        St, Sv = torch.as_tensor(Str), torch.as_tensor(Sval)
        Tt = torch.as_tensor(np.asarray(T), dtype=torch.float32)
        self.model = SmallCNN(Xt.shape[1])
        self._fit(Xt, St, Xv, Sv, Tt, rng)
        return self

    def _fit(self, Xt, St, Xv, Sv, T, rng):
        self._train(self.model, Xt, St, Xv, Sv, T, self.loss_fn, T, self.epochs, self.lr, rng)

    def predict_proba(self, X):
        return F.softmax(self._logits(torch.as_tensor(X)), 1).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)
