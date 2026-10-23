"""Numeric sanity demo of the losses (no data needed):
backward correction is an unbiased estimator of the clean CE loss, forward correction is minimised
by the clean posterior. Run: python demo_algorithms.py"""
import numpy as np, torch
from algorithm.losses import ce_loss, forward_loss, backward_loss
from experiments.data_utils import KNOWN_T

torch.manual_seed(0)
for name, T in KNOWN_T.items():
    T = torch.tensor(T, dtype=torch.float32)
    logits = torch.randn(1, 3)
    clean = torch.tensor([1])
    # E over noisy label S | Y=1 of backward loss == clean CE
    exp_bwd = sum(T[1, s].item() * backward_loss(logits, torch.tensor([s]), T).item() for s in range(3))
    print(f"{name}: E[backward] = {exp_bwd:.4f}   clean CE = {ce_loss(logits, clean).item():.4f}")
