import numpy as np, torch
from algorithm.losses import ce_loss, forward_loss, backward_loss, reweight_loss
from experiments.data_utils import KNOWN_T


def _T(name):
    return torch.tensor(KNOWN_T[name], dtype=torch.float32)


def test_known_T_are_row_stochastic_and_invertible():
    for T in KNOWN_T.values():
        assert np.allclose(T.sum(1), 1.0)
        assert abs(np.linalg.det(T)) > 1e-3


def test_forward_with_identity_equals_ce():
    logits, s = torch.randn(16, 3), torch.randint(0, 3, (16,))
    assert torch.allclose(forward_loss(logits, s, torch.eye(3)), ce_loss(logits, s), atol=1e-5)


def test_backward_with_identity_equals_ce():
    logits, s = torch.randn(16, 3), torch.randint(0, 3, (16,))
    assert torch.allclose(backward_loss(logits, s, torch.eye(3)), ce_loss(logits, s), atol=1e-5)


def test_backward_is_unbiased_estimator_of_clean_ce():
    for name in KNOWN_T:
        T = _T(name)
        logits = torch.randn(1, 3)
        for y in range(3):
            exp = sum(T[y, s].item() * backward_loss(logits, torch.tensor([s]), T).item() for s in range(3))
            assert abs(exp - ce_loss(logits, torch.tensor([y])).item()) < 1e-4


def test_reweight_with_identity_equals_ce():
    logits, s = torch.randn(16, 3), torch.randint(0, 3, (16,))
    assert torch.allclose(reweight_loss(logits, s, torch.eye(3)), ce_loss(logits, s), atol=1e-5)


def test_reweight_beta_is_detached_by_default():
    logits = torch.randn(8, 3, requires_grad=True)
    T = _T("FashionMNIST0.3").clone().requires_grad_(True)
    reweight_loss(logits, torch.randint(0, 3, (8,)), T).backward()
    assert T.grad is None or float(T.grad.abs().sum()) == 0.0


def test_reweight_beta_gives_T_gradient_when_not_detached():
    logits = torch.randn(8, 3)
    T = _T("FashionMNIST0.3").clone().requires_grad_(True)
    reweight_loss(logits, torch.randint(0, 3, (8,)), T, detach_beta=False).backward()
    assert float(T.grad.abs().sum()) > 0.0
