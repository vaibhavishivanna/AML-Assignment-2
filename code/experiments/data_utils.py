"""Data preparation: loading, preprocessing, random 80/20 split, given transition matrices.

Preprocessing (justified in the report):
  * uint8 pixels -> float32 in [0, 1]
  * FashionMNIST: 784 vector -> 1x28x28; CIFAR: HxWxC -> CxHxW
  * per-channel standardisation with mean/std computed on the TRAINING split only
    (validation and test images are transformed with the same statistics -> no leakage)
  * no augmentation, no label cleaning (labels are used exactly as given)
"""
import os
import numpy as np

# T[i, j] = P(noisy label = j | clean label = i)
KNOWN_T = {
    "FashionMNIST0.3": np.array([[0.7, 0.3, 0.0], [0.0, 0.7, 0.3], [0.3, 0.0, 0.7]]),
    "FashionMNIST0.6": np.array([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]]),
}
DATASETS = ["FashionMNIST0.3", "FashionMNIST0.6", "CIFAR"]


def dataset_name(path):
    return os.path.basename(path).replace(".npz", "")


def _to_nchw(X):
    X = X.astype(np.float32) / 255.0
    if X.ndim == 2:                       # flattened square image
        side = int(round(np.sqrt(X.shape[1])))
        X = X.reshape(-1, side, side)
    if X.ndim == 3:                       # (n, H, W) grayscale
        X = X[:, None]
    elif X.ndim == 4:                     # (n, H, W, C) -> (n, C, H, W)
        X = X.transpose(0, 3, 1, 2)
    return np.ascontiguousarray(X)


def load_dataset(path):
    """Returns Xtr, Str, Xts, Yts. Images float32 in [0,1], shape (n, C, H, W); labels int64."""
    d = np.load(path)
    return (_to_nchw(d["Xtr"]), d["Str"].astype(np.int64),
            _to_nchw(d["Xts"]), d["Yts"].astype(np.int64))


def split_train_val(X, S, seed, frac=0.8):
    """Independent random 80/20 split of the noisy data (different `seed` -> different split)."""
    idx = np.random.RandomState(seed).permutation(len(X))
    k = int(frac * len(X))
    tr, va = idx[:k], idx[k:]
    return X[tr], S[tr], X[va], S[va]


def standardize(Xtrain, *others):
    """Per-channel standardisation using training statistics only. Returns (Xtrain, *others)."""
    mean = Xtrain.mean((0, 2, 3), keepdims=True)
    std = Xtrain.std((0, 2, 3), keepdims=True) + 1e-6
    out = [((X - mean) / std).astype(np.float32) for X in (Xtrain, *others)]
    return tuple(out)


def describe_dataset(path):
    """Summary dict used by the data figures / report tables."""
    Xtr, Str, Xts, Yts = load_dataset(path)
    return {"name": dataset_name(path), "n_train_val": len(Xtr), "n_test": len(Xts),
            "image_shape": tuple(Xtr.shape[1:]),
            "noisy_label_freq": np.bincount(Str, minlength=3) / len(Str),
            "test_label_freq": np.bincount(Yts, minlength=3) / len(Yts),
            "pixel_range": (float(Xtr.min()), float(Xtr.max()))}
