import numpy as np
from algorithm import METHODS
from experiments.data_utils import KNOWN_T


def test_all_classifiers_fit_and_predict_on_tiny_data():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 1, 28, 28).astype(np.float32)
    S = rng.randint(0, 3, 120)
    T = KNOWN_T["FashionMNIST0.3"]
    for name, cls in METHODS.items():
        kw = {"epochs": 1}
        if name == "T-Revision":
            kw.update(warm=1, rev_epochs=1)
        clf = cls(**kw).fit(X[:96], S[:96], T, X[96:], S[96:], seed=0)
        pred = clf.predict(X)
        assert pred.shape == (120,) and set(pred) <= {0, 1, 2}
    # revised T stays a valid transition matrix
    assert np.allclose(clf.T_revised_.sum(1), 1.0, atol=1e-5) and (clf.T_revised_ >= 0).all()
