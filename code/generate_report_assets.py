"""Export the two report figures from one completed ten-run experiment.

No model fitting, copied report images or pandas are needed. Run from
code/: python generate_report_assets.py --results reproduced_results
The default destination is the repository's ASSETS/ directory.
"""
import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from transition_matrix import KNOWN_T, validate_transition
from paired_analysis import paired_statistics, holm_adjust


DATASETS = ("FashionMNIST0.3", "FashionMNIST0.6", "CIFAR")
N_RUNS = 10


def load_report_data(results):
    """Reject mixed settings, missing pairs and non-ten-anchor experiments."""
    results = Path(results)
    manifest_path = results / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    top_k = manifest["config"]["top_k"]
    if top_k != 10:
        raise ValueError("This report requires top_k=10; do not use an older experiment")
    inputs = [manifest_path]
    estimates, comparisons = {}, []
    reference_seeds = None
    for dataset in DATASETS:
        folders = sorted((results / dataset).glob("seed_*"),
                         key=lambda path: int(path.name.removeprefix("seed_")))
        if len(folders) != N_RUNS:
            raise ValueError(f"{dataset}: expected exactly {N_RUNS} completed seed folders")
        seeds = [int(folder.name.removeprefix("seed_")) for folder in folders]
        if len(set(seeds)) != N_RUNS:
            raise ValueError(f"{dataset}: duplicate seeds")
        if reference_seeds is not None and seeds != reference_seeds:
            raise ValueError("Datasets must share the same ten seed numbers")
        reference_seeds = seeds
        matrices, scores = [], {"Forward": [], "Reweight-R": []}
        mode = "estimated" if dataset == "CIFAR" else "known"
        for seed, folder in zip(seeds, folders):
            matrix_path = folder / "T_estimated.npy"
            anchor_path = folder / "anchor_train_indices.npy"
            matrix = validate_transition(np.load(matrix_path))
            anchors = np.load(anchor_path)
            if anchors.shape != (3, top_k):
                raise ValueError(f"{anchor_path}: anchors do not match top_k={top_k}")
            matrices.append(matrix)
            inputs.extend((matrix_path, anchor_path))
            for method in scores:
                record_path = folder / f"{mode}_{method}.json"
                record = json.loads(record_path.read_text())
                identity = (record["dataset"], record["seed"], record["mode"], record["method"])
                if identity != (dataset, seed, mode, method):
                    raise ValueError(f"{record_path}: result identity does not match folder")
                accuracy = float(record["accuracy"])
                if not np.isfinite(accuracy) or not 0 <= accuracy <= 100:
                    raise ValueError(f"{record_path}: invalid accuracy")
                scores[method].append(accuracy)
                inputs.append(record_path)
        values = np.stack(matrices)
        estimates[dataset] = values
        forward, reweight = (np.asarray(scores[m]) for m in ("Forward", "Reweight-R"))
        differences = reweight - forward
        comparisons.append({
            "dataset": dataset, "seeds": seeds, "differences_pp": differences.tolist(),
            **paired_statistics(forward, reweight),
        })
    for row, adjusted in zip(comparisons, holm_adjust([row["p_value"] for row in comparisons])):
        row["p_value_holm"] = adjusted
    return estimates, comparisons, inputs, manifest


def generate_report_assets(results, output):
    """Calculate every plotted value from per-run records and matrix arrays."""
    results, output = Path(results).resolve(), Path(output).resolve()
    estimates, comparisons, inputs, manifest = load_report_data(results)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aml2-assets-mpl"))
    cache = Path(tempfile.gettempdir()) / "aml2-assets-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300}):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.8), layout="constrained")
        for ax, row, color in zip(axes, comparisons, ("#2463a0", "#b65c16", "#378455")):
            ax.axhline(0, color=".45", linewidth=.7)
            ax.axhline(row["mean_difference_pp"], color=color, linestyle="--", linewidth=.9)
            ax.plot(row["seeds"], row["differences_pp"], "o-", color=color, linewidth=.8, markersize=3)
            ticks = [row["seeds"][i] for i in (0, 3, 6, 9)]
            ax.set(title=row["dataset"], xlabel="Seed", xticks=ticks)
            ax.tick_params(labelsize=8)
            note = (f"Mean {row['mean_difference_pp']:+.2f}\n"
                    f"CI [{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}]")
            ax.text(.03, .97, note, va="top", transform=ax.transAxes, fontsize=7.3,
                    bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 1})
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + (high - low) * .36)
        axes[0].set_ylabel("Reweight-R − Forward (pp)")
        fig.savefig(output / "paired_differences.png")
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(5.5, 4.2), layout="constrained")
        matrix_evidence = {}
        for i, dataset in enumerate(DATASETS[:2]):
            truth, values = KNOWN_T[dataset], estimates[dataset]
            mean = values.mean(axis=0)
            # Mean absolute error across runs, not absolute error of the mean.
            entry_errors = np.abs(values - truth).mean(axis=0)
            matrix_evidence[dataset] = {
                "n_runs": len(values), "mean_estimate": mean.tolist(),
                "mean_absolute_entrywise_error": entry_errors.tolist(),
                "mae_mean": float(np.abs(values - truth).mean()),
                "frobenius_mean": float(np.linalg.norm(values - truth, axis=(1, 2)).mean()),
            }
            panels = ((truth, "Given T"), (mean, "Mean estimate\n(k = 10)"),
                      (entry_errors, "Mean absolute\nentrywise error"))
            for j, (matrix, title) in enumerate(panels):
                ax = axes[i, j]
                maximum = 1 if j < 2 else .2
                im = ax.imshow(matrix, vmin=0, vmax=maximum,
                               cmap="Blues" if j < 2 else "Oranges")
                for clean in range(3):
                    for noisy in range(3):
                        ax.text(noisy, clean, f"{matrix[clean, noisy]:.3f}", ha="center",
                                va="center", fontsize=7.2,
                                color="white" if matrix[clean, noisy] > maximum * .55 else "black")
                ax.set(xticks=range(3), yticks=range(3), xlabel="Noisy class")
                ax.set_title(title, fontsize=8)
                label = f"FM{('0.3', '0.6')[i]}\nClean class" if j == 0 else "Clean class"
                ax.set_ylabel(label, fontsize=8)
                ax.tick_params(labelsize=7)
                bar = fig.colorbar(im, ax=ax, shrink=.8, fraction=.046, pad=.03)
                bar.ax.tick_params(labelsize=6)
        fig.savefig(output / "transition_estimation_comparison.png")
        plt.close(fig)

    with (output / "paired_comparison.csv").open("w", newline="") as stream:
        fields = ["dataset", "mean_difference_pp", "ci95_low_pp", "ci95_high_pp", "p_value", "p_value_holm"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in comparisons)
    hashes = {str(p.relative_to(results)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    evidence = {"results_directory": str(results), "experiment_config": manifest["config"],
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "paired_analysis_sha256": hashlib.sha256(
                    Path(__file__).with_name("paired_analysis.py").read_bytes()).hexdigest(),
                "input_sha256": hashes, "paired_comparisons": comparisons,
                "matrix_evidence": matrix_evidence,
                "notes": "No training performed. Matrices use clean rows/noisy columns. "
                         "Individual paired t intervals, df=9, sample SD; panels have different y scales."}
    (output / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (output / "README.txt").write_text(
        "REPORT IMAGES GENERATED BY THE REPOSITORY CODE\n\n"
        "Figure 1: paired_differences.png\n"
        "Figure 2: transition_estimation_comparison.png\n\n"
        "Copy these two PNGs into the report's images/ folder, replacing files with the same names.\n"
        "Keep the LaTeX paths images/paired_differences.png and "
        "images/transition_estimation_comparison.png.\n"
        "Do not substitute the older one-anchor heatmap with diagonal 0.846, 0.807, 0.899.\n"
        "This export uses the ten-anchor run: Figure 2's FM0.3 diagonal is 0.818, 0.791, 0.838.\n"
        "Figure 2's right column is mean ABSOLUTE ENTRYWISE ERROR, not standard deviation.\n"
        "All colour scales are displayed. No AI image generation or model retraining was used.\n\n"
        "From code/, regenerate from any complete matching run:\n"
        "python generate_report_assets.py --results reproduced_results --out ../ASSETS\n\n"
        "For this export, the source experiment was:\n" + str(results) + "\n\n"
        "provenance.json records actual arrays, settings and input hashes; "
        "paired_comparison.csv records the plotted differences and confidence intervals.\n")
    print(f"Saved Figure 1, Figure 2 and provenance to {output}")
    for dataset, row in matrix_evidence.items():
        diagonal = np.diag(row["mean_estimate"])
        print(f"{dataset}: mean estimated diagonal = {diagonal.round(6).tolist()}")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="One completed top-k=10 experiment with ten runs per dataset")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "ASSETS")
    args = parser.parse_args()
    generate_report_assets(args.results, args.out)
