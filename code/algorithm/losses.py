"""Loss functions. Convention: T[i, j] = P(noisy = j | clean = i); p = softmax(logits) ~ P(clean | x),
so the noisy posterior is p @ T."""
import torch
import torch.nn.functional as F

EPS = 1e-7


def ce_loss(logits, s, T=None):
    """Plain cross-entropy on the noisy labels (baseline)."""
    return F.cross_entropy(logits, s)


def forward_loss(logits, s, T):
    """Forward correction: -log (T^T p)_s."""
    p = F.softmax(logits, 1)
    return F.nll_loss(torch.log(p @ T + EPS), s)


def backward_loss(logits, s, T):
    """Backward correction: sum_j (T^-1)[s, j] * (-log p_j). Unbiased for the clean CE risk."""
    logp = F.log_softmax(logits, 1)
    Tinv = torch.linalg.inv(T)
    return -(Tinv[s] * logp).sum(1).mean()


def reweight_loss(logits, s, T, detach_beta=True):
    """Importance reweighting: beta * CE, beta = p_s / (T^T p)_s."""
    p = F.softmax(logits, 1)
    num = p.gather(1, s[:, None]).squeeze(1)
    den = (p @ T).gather(1, s[:, None]).squeeze(1)
    beta = num / (den + EPS)
    if detach_beta:
        beta = beta.detach()
    return (beta * F.cross_entropy(logits, s, reduction="none")).mean()
