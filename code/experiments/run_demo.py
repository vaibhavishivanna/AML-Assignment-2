"""Quick demo: every classifier on a 3000-image subset, 3 epochs (about 1 minute).
    python -m experiments.run_demo data/FashionMNIST0.3.npz"""
import sys
import numpy as np
from algorithm import METHODS
from evaluation import top1_accuracy
from experiments.data_utils import KNOWN_T, dataset_name, load_dataset, split_train_val, standardize

path = sys.argv[1] if len(sys.argv) > 1 else "data/FashionMNIST0.3.npz"
Xtr, Str, Xts, Yts = load_dataset(path)
T = KNOWN_T[dataset_name(path)]
Xt, St, Xv, Sv = split_train_val(Xtr[:3000], Str[:3000], seed=0)
Xt, Xv, Xs = standardize(Xt, Xv, Xts)
for name, cls in METHODS.items():
    clf = cls(epochs=3, rev_epochs=2).fit(Xt, St, T, Xv, Sv, seed=0) if name == "T-Revision" else cls(epochs=3).fit(Xt, St, T, Xv, Sv, seed=0)
    print(f"{name:11s} test acc = {top1_accuracy(clf.predict(Xs), Yts):.2f}%")
