"""Top-1 accuracy and aggregation of repeated runs (mean +- std) into CSV / LaTeX."""
import csv
import numpy as np


def top1_accuracy(pred, y):
    """number of correctly classified examples / total, in %."""
    return float((np.asarray(pred) == np.asarray(y)).mean() * 100)


def summarize(raw):
    """raw: {dataset: {method: [acc, ...]}} -> list of row dicts."""
    rows = []
    for ds, per_method in raw.items():
        for m, accs in per_method.items():
            rows.append({"dataset": ds, "method": m, "mean": float(np.mean(accs)),
                         "std": float(np.std(accs)), "n_runs": len(accs)})
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "mean", "std", "n_runs"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "mean": f"{r['mean']:.2f}", "std": f"{r['std']:.2f}"})


def write_latex_table(rows, path, caption="Test accuracy (\\%), mean $\\pm$ std over random 80/20 splits.",
                      label="tab:accuracy"):
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    cell = {(r["dataset"], r["method"]): r for r in rows}
    best = {d: max((cell[(d, m)]["mean"] for m in methods if (d, m) in cell)) for d in datasets}
    lines = ["\\begin{table}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\begin{center}",
             "\\begin{tabular}{l" + "c" * len(datasets) + "}", "\\toprule",
             "\\textbf{Method} & " + " & ".join(f"\\textbf{{{d}}}" for d in datasets) + " \\\\", "\\midrule"]
    for m in methods:
        cells = []
        for d in datasets:
            r = cell.get((d, m))
            if r is None:
                cells.append("--")
                continue
            s = f"{r['mean']:.2f} $\\pm$ {r['std']:.2f}"
            cells.append(f"\\textbf{{{s}}}" if abs(r["mean"] - best[d]) < 1e-9 else s)
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}", "\\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")
