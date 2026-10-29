"""Reproduce paired confidence intervals, p-values and Holm adjustment.

Example: python compare_runs.py --results reproduced_results
Reads per-run JSON evidence from the selected experiment, never a historical
hard-coded directory. Figure generation is handled by generate_report_assets.py.
"""
import argparse
import csv
import json
from pathlib import Path

from paired_analysis import analyze_saved_runs


def compare_runs(results, output=None):
    results = Path(results)
    rows = analyze_saved_runs(results)
    output = Path(output) if output is not None else results / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "mode", "n_runs", "forward_mean", "forward_std", "reweight_mean",
              "reweight_std", "mean_difference_pp", "difference_std_sample", "ci95_low_pp",
              "ci95_high_pp", "p_value", "p_value_holm", "forward_wins", "reweight_wins", "ties"]
    with (output / "paired_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    with (output / "paired_per_run.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "mode", "seed", "forward_accuracy", "reweight_accuracy", "difference_pp"])
        for row in rows:
            for values in zip(row["seeds"], row["forward"], row["reweight"], row["differences_pp"]):
                writer.writerow([row["dataset"], row["mode"], *values])
    evidence = {"source_results": str(results.resolve()), "comparisons": rows,
                "inference": "Two-sided paired t; sample SD; individual 95% CIs; "
                             "Holm adjustment over all dataset/mode comparisons in this export",
                "descriptive_sd_ddof": 0, "paired_sd_ddof": 1}
    (output / "paired_analysis.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    lines = [r"\begin{tabular}{llrrrr}",
             r"Dataset & Matrix & Difference (pp) & 95\% CI & $p$ & Holm $p$ \\", r"\hline"]
    for row in rows:
        interval = f"[{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}]"
        lines.append(f"{row['dataset']} & {row['mode']} & {row['mean_difference_pp']:+.2f} & "
                     f"{interval} & {row['p_value']:.5f} & {row['p_value_holm']:.5f} " + r"\\")
        print(f"{row['dataset']} ({row['mode']}): {row['mean_difference_pp']:+.2f} pp, "
              f"95% CI {interval}, p={row['p_value']:.6f}, Holm p={row['p_value_holm']:.6f}")
    (output / "paired_comparison.tex").write_text("\n".join(lines + [r"\end{tabular}"]) + "\n")
    print(f"Saved paired analysis to {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Completed experiment containing per-seed classifier JSON records")
    parser.add_argument("--out", type=Path, help="Default: analysis/ inside --results")
    args = parser.parse_args()
    compare_runs(args.results, args.out)
