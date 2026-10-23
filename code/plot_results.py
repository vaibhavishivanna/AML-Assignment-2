"""Results figures from final_results/summary.csv:
   figures/accuracy_comparison.png  grouped bars (mean test accuracy) + table underneath
   figures/accuracy_errorbars.png   zoomed mean +- std plot, makes small differences visible"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASTEL = {"CE": "#8EC9EE", "Forward": "#8CC63F", "Backward": "#AB8FDD", "T-Revision": "#F5B041"}
DARK = {"CE": "#3B8FC4", "Forward": "#5E8F1B", "Backward": "#7A5CC0", "T-Revision": "#C27F0E"}
MARK = {"CE": "s", "Forward": "o", "Backward": "^", "T-Revision": "D"}
res = sys.argv[1] if len(sys.argv) > 1 else "final_results"
rows = list(csv.DictReader(open(os.path.join(res, "summary.csv"))))
datasets = list(dict.fromkeys(r["dataset"] for r in rows))
methods = list(dict.fromkeys(r["method"] for r in rows))
get = lambda d, m: next((r for r in rows if r["dataset"] == d and r["method"] == m), None)
out_dir = os.path.join(res, "figures"); os.makedirs(out_dir, exist_ok=True)

# ---------- 1. bars + table (style of the reference report figure)
fig = plt.figure(figsize=(8.2, 5.6))
ax = fig.add_axes([0.09, 0.40, 0.88, 0.50])
w = 0.8 / len(methods)
for k, m in enumerate(methods):
    xs, mu, sd = [], [], []
    for i, d in enumerate(datasets):
        r = get(d, m)
        if r:
            xs.append(i + (k - (len(methods) - 1) / 2) * w); mu.append(float(r["mean"])); sd.append(float(r["std"]))
    ax.bar(xs, mu, w * 0.92, yerr=sd, capsize=3, color=PASTEL[m], edgecolor="#444", lw=0.8,
           error_kw={"lw": 1, "ecolor": "#333"}, label=m, zorder=3)
    for x, y in zip(xs, mu):
        ax.text(x, y + 1.0, f"{y:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax.set_ylim(0, 112); ax.set_xlim(-0.5, len(datasets) - 0.5)
ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Percentage", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=9)
ax.set_xlabel("Datasets", fontsize=9, fontweight="bold")
ax.yaxis.grid(True, ls=":", color="#999", zorder=0); ax.set_axisbelow(True)
ax.set_title("Comparison of Test Accuracy (mean over random 80/20 splits)", fontsize=10, fontweight="bold")
ax.legend(loc="lower left", fontsize=8, framealpha=0.95, edgecolor="#888")
cell = [[(f"{float(get(d, m)['mean']):.2f} ± {float(get(d, m)['std']):.2f}" if get(d, m) else "--") for d in datasets] for m in methods]
tb = fig.add_axes([0.17, 0.04, 0.80, 0.26]); tb.axis("off")
t = tb.table(cellText=cell, rowLabels=methods, colLabels=datasets, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.5)
for (r, c), cl in t.get_celld().items():
    cl.set_edgecolor("#666"); cl.set_linewidth(0.6)
    if r == 0 and c >= 0: cl.set_facecolor("#E4F1FA"); cl.set_text_props(fontweight="bold")
    if c == -1 and r > 0: cl.set_facecolor(PASTEL[methods[r - 1]]); cl.set_text_props(fontweight="bold")
fig.savefig(os.path.join(out_dir, "accuracy_comparison.png"), dpi=200); plt.close(fig)

# ---------- 2. zoomed error-bar plot
fig, ax = plt.subplots(figsize=(1.9 * len(datasets) + 2.8, 3.8)); step = 0.17
for k, m in enumerate(methods):
    xs, mu, sd = [], [], []
    for i, d in enumerate(datasets):
        r = get(d, m)
        if r:
            xs.append(i + (k - (len(methods) - 1) / 2) * step); mu.append(float(r["mean"])); sd.append(float(r["std"]))
    ax.errorbar(xs, mu, yerr=sd, fmt=MARK[m], ms=7, capsize=4, lw=1.6, color=DARK[m], mfc=PASTEL[m], label=m, zorder=3)
    for x, y, s in zip(xs, mu, sd):
        ax.annotate(f"{y:.2f}", (x, y + s), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7, color=DARK[m])
for i in range(0, len(datasets), 2):
    ax.axvspan(i - 0.5, i + 0.5, color="#EEF6FC", zorder=0)
ax.set_xlim(-0.5, len(datasets) - 0.5); ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, fontsize=9)
ax.set_ylabel("test accuracy (%)", fontsize=9); ax.yaxis.grid(True, color="#DDD", lw=0.7); ax.set_axisbelow(True)
ax.legend(fontsize=8, frameon=False, ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig(os.path.join(out_dir, "accuracy_errorbars.png"), dpi=200)
print(os.path.join(out_dir, "accuracy_comparison.png"))
