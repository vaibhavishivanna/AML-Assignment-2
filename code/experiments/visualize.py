"""Data-preparation figures: sample images, noisy-label distribution, given transition matrices."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import KNOWN_T, load_dataset, dataset_name

NOISY, CLEAN, GREY = "#D55E00", "#009E73", "#666666"   # vermillion / green: colour-blind safe, no blue


def plot_samples(path, out_dir, per_class=8, seed=0):
    """Rows = NOISY label 0/1/2, columns = random training images carrying that label."""
    Xtr, Str, _, _ = load_dataset(path)
    rng = np.random.RandomState(seed)
    fig, axes = plt.subplots(3, per_class, figsize=(per_class * 1.1, 3.6))
    for c in range(3):
        idx = rng.choice(np.where(Str == c)[0], per_class, replace=False)
        for j, i in enumerate(idx):
            img = Xtr[i].transpose(1, 2, 0)
            axes[c, j].imshow(img.squeeze(), cmap="gray" if img.shape[2] == 1 else None)
            axes[c, j].axis("off")
        axes[c, 0].set_title(f"noisy label {c}", fontsize=9, loc="left")
    fig.suptitle(f"{dataset_name(path)}: training images grouped by noisy label", fontsize=10)
    fig.tight_layout()
    f = os.path.join(out_dir, f"samples_{dataset_name(path)}.png")
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def plot_label_distribution(paths, out_dir):
    """Share of each class in the noisy train/val labels vs the clean test labels."""
    fig, axes = plt.subplots(1, len(paths), figsize=(3.2 * len(paths), 2.8), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), paths):
        _, S, _, Y = load_dataset(p)
        x = np.arange(3)
        ax.bar(x - 0.2, np.bincount(S, minlength=3) / len(S), 0.4, color=NOISY, label="noisy train/val")
        ax.bar(x + 0.2, np.bincount(Y, minlength=3) / len(Y), 0.4, color=CLEAN, label="clean test")
        ax.set_xticks(x); ax.set_title(dataset_name(p), fontsize=9)
        ax.set_xlabel("class"); ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("share of examples")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    f = os.path.join(out_dir, "label_distribution.png")
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def plot_transition_matrices(out_dir, extra=None):
    """Heatmaps of the given T (and optionally an estimated T: {'CIFAR (estimated)': matrix})."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("lightblue", ["#EAF4FB", "#A8D4F0", "#5AA9DE"])
    mats = dict(KNOWN_T)
    mats.update(extra or {})
    fig, axes = plt.subplots(1, len(mats), figsize=(3.4 * len(mats), 3.2))
    for ax, (name, T) in zip(np.atleast_1d(axes), mats.items()):
        im = ax.imshow(T, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="#1F3A4D")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Noisy label", fontweight="bold"); ax.set_ylabel("Clean label", fontweight="bold")
        ax.set_title(f"Transition matrix: {name}", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    f = os.path.join(out_dir, "transition_matrices.png")
    fig.savefig(f, dpi=150); plt.close(fig)
    return f


def make_data_figures(data_dir="data", out_dir="final_results/figures"):
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(data_dir, n + ".npz") for n in ("FashionMNIST0.3", "FashionMNIST0.6", "CIFAR")]
    paths = [p for p in paths if os.path.exists(p)]
    out = [plot_samples(p, out_dir) for p in paths]
    out += [plot_label_distribution(paths, out_dir), plot_transition_matrices(out_dir)]
    return out


if __name__ == "__main__":
    print("\n".join(make_data_figures()))
