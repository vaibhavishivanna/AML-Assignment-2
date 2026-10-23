"""Run all classifiers on the datasets with >= 10 random 80/20 splits. Resumable.

    python run_experiments.py                       # FashionMNIST0.3 + 0.6 (given T)
    python run_experiments.py --datasets CIFAR --T_file estimated_T.npy
Reads data/<name>.npz, writes final_results/{raw_runs.json, summary.csv, accuracy_summary_table.tex}.
"""
import argparse, json, os, time
import numpy as np

from algorithm import METHODS
from evaluation import top1_accuracy, summarize, write_csv, write_latex_table
from experiments.data_utils import KNOWN_T, load_dataset, split_train_val, standardize

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["FashionMNIST0.3", "FashionMNIST0.6"])
ap.add_argument("--methods", nargs="+", default=list(METHODS))
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--T_file", default=None, help=".npy 3x3 matrix, required for CIFAR")
ap.add_argument("--data_dir", default="data")
ap.add_argument("--out", default="final_results")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
raw_path = os.path.join(a.out, "raw_runs.json")
raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}

for ds in a.datasets:
    T = np.load(a.T_file) if (a.T_file and ds == "CIFAR") else KNOWN_T[ds]
    Xtr, Str, Xts, Yts = load_dataset(os.path.join(a.data_dir, ds + ".npz"))
    rec = raw.setdefault(ds, {m: [] for m in a.methods})
    for r in range(a.runs):
        Xt, St, Xv, Sv = split_train_val(Xtr, Str, seed=r)
        Xt, Xv, Xs = standardize(Xt, Xv, Xts)
        for m in a.methods:
            accs = rec.setdefault(m, [])
            if len(accs) > r:
                continue                       # already done (resume)
            t0 = time.time()
            clf = METHODS[m](epochs=a.epochs).fit(Xt, St, T, Xv, Sv, seed=r)
            accs.append(top1_accuracy(clf.predict(Xs), Yts))
            print(f"{ds} run{r} {m:11s} acc={accs[-1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
            json.dump(raw, open(raw_path, "w"), indent=1)
        # keep summary files fresh after every completed run
        rows = summarize(raw)
        write_csv(rows, os.path.join(a.out, "summary.csv"))
        write_latex_table(rows, os.path.join(a.out, "accuracy_summary_table.tex"))
for r in summarize(raw):
    print(f"{r['dataset']:16s} {r['method']:11s} {r['mean']:.2f} +- {r['std']:.2f}  (n={r['n_runs']})")
