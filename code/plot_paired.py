"""Per-seed paired differences vs the cross-entropy baseline -> final_results/figures/paired_differences.png"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"Forward": "#5E8F1B", "Backward": "#7A5CC0", "T-Revision": "#C27F0E"}
FILL = {"Forward": "#8CC63F", "Backward": "#AB8FDD", "T-Revision": "#F5B041"}
res = sys.argv[1] if len(sys.argv) > 1 else "final_results"
raw = json.load(open(os.path.join(res, "raw_runs.json")))
fig, axes = plt.subplots(1, len(raw), figsize=(3.4 * len(raw), 2.9), sharey=True)
rng = np.random.RandomState(0)
for ax, (ds, d) in zip(np.atleast_1d(axes), raw.items()):
    ax.axhline(0, color="#444", lw=1)
    for k, m in enumerate(["Forward", "Backward", "T-Revision"]):
        diff = np.array(d[m]) - np.array(d["CE"])
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(diff)), diff, s=26, color=FILL[m], edgecolor=COL[m], zorder=3)
        ax.hlines(diff.mean(), k - 0.28, k + 0.28, color=COL[m], lw=2.5, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Forward", "Backward", "T-Revision"], fontsize=8)
    ax.set_title(ds, fontsize=9, fontweight="bold"); ax.yaxis.grid(True, color="#DDD", lw=0.7); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("accuracy gain over CE (points)", fontsize=8)
fig.tight_layout()
out = os.path.join(res, "figures", "paired_differences.png")
fig.savefig(out, dpi=200); print(out)
