"""Build result tables and plots; standard deviations use ddof=0."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from experiments.data_utils import KNOWN_T


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records, keys, fields):
    groups = {}
    for record in records:
        group = tuple(record[k] for k in keys)
        groups.setdefault(group, []).append(record)
    rows = []
    for key, values in sorted(groups.items()):
        row = dict(zip(keys, key))
        row["n_runs"] = len(values)
        for field in fields:
            numbers = [r[field] for r in values]
            row[field + "_mean"] = float(np.mean(numbers))
            row[field + "_std"] = float(np.std(numbers, ddof=0))
        rows.append(row)
    return rows


def summarize_results(output):
    output = Path(output)
    records, errors, matrices, revisions, revised_matrices = [], [], {}, [], {}
    for path in sorted(output.glob("*/seed_*/estimator.json")):
        dataset, seed = path.parent.parent.name, int(path.parent.name.split("_")[-1])
        info = json.loads(path.read_text())
        T = np.load(path.parent / "T_estimated.npy")
        matrices.setdefault(dataset, []).append(T)
        if "errors" in info:
            errors.append({"dataset": dataset, "seed": seed, **info["errors"]})
        for method_path in sorted(path.parent.glob("*_*R.json")) + sorted(path.parent.glob("*_Forward.json")):
            record = json.loads(method_path.read_text())
            records.append(record)
            if record["method"] == "Reweight-R":
                revised_path = method_path.with_name(method_path.stem + "_T_revised.npy")
                revised_matrices.setdefault((dataset, record["mode"]), []).append(np.load(revised_path))
                row = {k: record[k] for k in ["dataset", "seed", "mode", "accuracy", "accuracy_before_revision",
                                            "revision_gain_pp", "matrix_change_frobenius"]}
                if "revised_matrix_errors" in record:
                    row.update(input_frobenius=record["input_matrix_errors"]["frobenius"],
                               revised_frobenius=record["revised_matrix_errors"]["frobenius"])
                revisions.append(row)
    write_csv(output / "estimator_errors_per_run.csv", errors)
    error_summary = aggregate(errors, ["dataset"], ["mae", "frobenius", "relative_l1"])
    write_csv(output / "estimator_error_summary.csv", error_summary)
    # Mean matrices are for reporting, not per-split training.
    for dataset, values in matrices.items():
        np.save(output / f"{dataset}_T_mean.npy", np.mean(values, axis=0))
        np.save(output / f"{dataset}_T_std.npy", np.std(values, axis=0, ddof=0))
    accuracy = aggregate(records, ["dataset", "mode", "method"], ["accuracy"])
    write_csv(output / "accuracy_summary.csv", accuracy)
    write_csv(output / "accuracy_per_run.csv", records_for_csv(records))
    # CIFAR has no ground-truth matrix errors.
    if revisions:
        fields = list(dict.fromkeys(k for r in revisions for k in r))
        write_csv(output / "revision_per_run.csv", [{k: r.get(k, "") for k in fields} for r in revisions])
    revision_summary = aggregate(revisions, ["dataset", "mode"],
                                 ["accuracy", "accuracy_before_revision", "revision_gain_pp", "matrix_change_frobenius"])
    for row in revision_summary:
        group = [r for r in revisions if (r["dataset"], r["mode"]) == (row["dataset"], row["mode"])]
        row.update(improved=sum(r["revision_gain_pp"] > 1e-8 for r in group),
                   worsened=sum(r["revision_gain_pp"] < -1e-8 for r in group),
                   unchanged=sum(abs(r["revision_gain_pp"]) <= 1e-8 for r in group))
    for (dataset, mode), values in revised_matrices.items():
        np.save(output / f"{dataset}_{mode}_T_revised_mean.npy", np.mean(values, axis=0))
        np.save(output / f"{dataset}_{mode}_T_revised_std.npy", np.std(values, axis=0))
    matrix_errors = aggregate([r for r in revisions if "input_frobenius" in r],
                              ["dataset", "mode"], ["input_frobenius", "revised_frobenius"])
    write_csv(output / "revision_matrix_errors.csv", matrix_errors)
    write_csv(output / "revision_summary.csv", revision_summary)
    if revision_summary:
        lines = [r"\begin{tabular}{llrccc}",
                 r"Dataset & Matrix & Runs & Before (\%) & After (\%) & Gain (pp) \\", r"\hline"]
        for r in revision_summary:
            cells = [f"{r[k+'_mean']:.2f} $\\pm$ {r[k+'_std']:.2f}" for k in
                     ["accuracy_before_revision", "accuracy", "revision_gain_pp"]]
            lines.append(f"{r['dataset']} & {r['mode']} & {r['n_runs']} & " + " & ".join(cells) + r"\\")
        (output / "revision_table.tex").write_text("\n".join(lines + [r"\end{tabular}"]) + "\n")
    latex = [r"% Generated from actual completed runs. Standard deviations use ddof=0."]
    if accuracy:
        latex += [r"\begin{tabular}{lllrc}", r"Dataset & Matrix & Method & Runs & Accuracy (\%) \\", r"\hline"]
        for r in accuracy:
            latex.append(f"{r['dataset']} & {r['mode']} & {r['method']} & {r['n_runs']} & "
                         f"{r['accuracy_mean']:.2f} $\\pm$ {r['accuracy_std']:.2f} " + r"\\")
        latex += [r"\end{tabular}"]
        (output / "accuracy_table.tex").write_text("\n".join(latex) + "\n")
    plot_results(output, matrices, accuracy, revision_summary)
    print(f"Saved summaries: {len(matrices)} datasets, {len(records)} classifier runs -> {output}", flush=True)
    return {"estimator": error_summary, "accuracy": accuracy, "revision": revision_summary}


def records_for_csv(records):
    keys = ["dataset", "seed", "mode", "method", "accuracy", "n_test", "seconds"]
    return [{k: r[k] for k in keys} for r in records]


def plot_results(output, matrices, accuracy, revisions):
    # Use a writable plot cache.
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    if matrices:
        fig, axes = plt.subplots(len(matrices), 3, figsize=(9, 2.8 * len(matrices)), squeeze=False)
        for row, (dataset, values) in enumerate(sorted(matrices.items())):
            panels = [("Given T", KNOWN_T.get(dataset)),
                      (f"Estimated mean (n={len(values)})", np.mean(values, axis=0)),
                      ("Estimate SD", np.std(values, axis=0))]
            for ax, (title, matrix) in zip(axes[row], panels):
                ax.set_title(f"{dataset}\n{title}", fontsize=9)
                if matrix is None:
                    ax.text(.5, .5, "Unknown ground truth", ha="center", va="center")
                    ax.axis("off")
                    continue
                ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                                color="white" if matrix[i,j] > .6 else "black", fontsize=9)
                ax.set_xticks(range(3)); ax.set_yticks(range(3))
                ax.set_xlabel("Noisy class"); ax.set_ylabel("Clean class")
        fig.tight_layout()
        fig.savefig(figures / "initial_matrices.png", dpi=180)
        plt.close(fig)
    if accuracy:
        groups = list(dict.fromkeys((r["dataset"], r["mode"]) for r in accuracy))
        fig, ax = plt.subplots(figsize=(max(7, 2 * len(groups)), 4))
        for offset, method in [(-.12, "Forward"), (.12, "Reweight-R")]:
            rows = [next((r for r in accuracy if (r["dataset"], r["mode"]) == g and r["method"] == method), None)
                    for g in groups]
            valid = [(i, r) for i, r in enumerate(rows) if r]
            ax.errorbar([i + offset for i, _ in valid], [r["accuracy_mean"] for _, r in valid],
                        yerr=[r["accuracy_std"] for _, r in valid], fmt="o", capsize=4, label=method)
        ax.set_xticks(range(len(groups)), [f"{d}\n{m}" for d, m in groups])
        ax.set_ylabel("Clean test accuracy (%) — mean ± SD")
        ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
        fig.savefig(figures / "accuracy_comparison.png", dpi=180); plt.close(fig)
    if revisions:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0, color="gray", linewidth=1)
        ax.errorbar(range(len(revisions)), [r["revision_gain_pp_mean"] for r in revisions],
                    yerr=[r["revision_gain_pp_std"] for r in revisions], fmt="o", capsize=4)
        ax.set_xticks(range(len(revisions)), [f"{r['dataset']}\n{r['mode']}" for r in revisions])
        ax.set_ylabel("Paired accuracy gain after revision (percentage points)")
        fig.tight_layout(); fig.savefig(figures / "revision_gain.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path(__file__).resolve().parent / "reproduced_results")
    summarize_results(parser.parse_args().output)
