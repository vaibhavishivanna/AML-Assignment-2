import numpy as np
from experiments.data_utils import split_train_val, standardize, _to_nchw


def test_split_is_80_20_disjoint_and_seeded():
    X, S = np.arange(1000, dtype=np.float32).reshape(1000, 1, 1, 1), np.zeros(1000, dtype=np.int64)
    Xt, _, Xv, _ = split_train_val(X, S, seed=3)
    assert len(Xt) == 800 and len(Xv) == 200
    assert not set(Xt.ravel()) & set(Xv.ravel())
    assert np.array_equal(Xt, split_train_val(X, S, seed=3)[0])
    assert not np.array_equal(Xt, split_train_val(X, S, seed=4)[0])


def test_standardize_uses_train_statistics_only():
    rng = np.random.RandomState(0)
    Xt = rng.rand(200, 3, 4, 4).astype(np.float32)
    Xo = rng.rand(50, 3, 4, 4).astype(np.float32) + 1.0
    A, B = standardize(Xt, Xo)
    assert np.allclose(A.mean((0, 2, 3)), 0, atol=1e-4) and np.allclose(A.std((0, 2, 3)), 1, atol=1e-3)
    assert B.mean() > 0.5                      # other split is NOT re-centred


def test_shapes():
    assert _to_nchw(np.zeros((5, 784), dtype=np.uint8)).shape == (5, 1, 28, 28)
    assert _to_nchw(np.zeros((5, 32, 32, 3), dtype=np.uint8)).shape == (5, 3, 32, 32)
    assert _to_nchw(np.full((2, 784), 255, dtype=np.uint8)).max() == 1.0
