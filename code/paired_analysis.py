"""Shared paired statistics for the report and its figure exporter."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t, ttest_1samp


def paired_statistics(forward, reweight):
    """Use sample SD for t inference and population SD for descriptive results."""
    forward, reweight = np.asarray(forward, dtype=float), np.asarray(reweight, dtype=float)
    if forward.ndim != 1 or forward.shape != reweight.shape or forward.size < 2:
        raise ValueError("Expected at least two paired accuracies of equal length")
    if not np.isfinite(np.r_[forward, reweight]).all():
        raise ValueError("Accuracies must be finite")
    if ((np.r_[forward, reweight] < 0) | (np.r_[forward, reweight] > 100)).any():
        raise ValueError("Accuracies must be between 0 and 100")
    differences = reweight - forward
    mean, sample_sd = float(differences.mean()), float(differences.std(ddof=1))
    half_width = float(t.ppf(.975, len(differences) - 1) * sample_sd / np.sqrt(len(differences)))
    # Handle degenerate pairs explicitly instead of exporting scipy's NaN.
    p_value = (1.0 if mean == 0 else 0.0) if sample_sd == 0 else float(ttest_1samp(differences, 0).pvalue)
    return {
        "n_runs": len(differences),
        "forward_mean": float(forward.mean()), "forward_std": float(forward.std(ddof=0)),
        "reweight_mean": float(reweight.mean()), "reweight_std": float(reweight.std(ddof=0)),
        "mean_difference_pp": mean, "difference_std_sample": sample_sd,
        "ci95_low_pp": mean - half_width, "ci95_high_pp": mean + half_width,
        "p_value": p_value,
        "forward_wins": int((differences < -1e-8).sum()),
        "reweight_wins": int((differences > 1e-8).sum()),
        "ties": int((np.abs(differences) <= 1e-8).sum()),
    }


def holm_adjust(p_values):
    """Holm step-down adjustment, retaining the original comparison order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Expected finite p-values between 0 and 1")
    adjusted, previous = np.empty_like(values), 0.0
    for rank, index in enumerate(np.argsort(values, kind="stable")):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted.tolist()


def analyze_saved_runs(results):
    """Pair actual JSON records by dataset, matrix mode and seed; never drop pairs."""
    results = Path(results)
    groups = {}
    paths = sorted(results.glob("*/seed_*/*_Forward.json"))
    paths += sorted(results.glob("*/seed_*/*_Reweight-R.json"))
    if not paths:
        raise ValueError(f"No completed classifier records in {results}")
    for path in paths:
        record = json.loads(path.read_text())
        dataset, seed = path.parent.parent.name, int(path.parent.name.removeprefix("seed_"))
        method, mode = record["method"], record["mode"]
        if (record["dataset"], record["seed"]) != (dataset, seed):
            raise ValueError(f"{path}: record identity does not match folder")
        if method not in ("Forward", "Reweight-R") or mode not in ("known", "estimated"):
            raise ValueError(f"{path}: unsupported classifier or matrix mode")
        if path.name != f"{mode}_{method}.json":
            raise ValueError(f"{path}: filename does not match record")
        pair = groups.setdefault((dataset, mode), {}).setdefault(seed, {})
        if method in pair:
            raise ValueError(f"Duplicate {dataset}/{mode}/{seed}/{method}")
        pair[method] = float(record["accuracy"])
    rows = []
    for (dataset, mode), pairs in sorted(groups.items()):
        seeds = sorted(pairs)
        for seed in seeds:
            if set(pairs[seed]) != {"Forward", "Reweight-R"}:
                raise ValueError(f"{dataset}/{mode}/seed_{seed:02d}: missing paired classifier")
        forward = [pairs[seed]["Forward"] for seed in seeds]
        reweight = [pairs[seed]["Reweight-R"] for seed in seeds]
        rows.append({"dataset": dataset, "mode": mode, **paired_statistics(forward, reweight),
                     "seeds": seeds, "forward": forward, "reweight": reweight,
                     "differences_pp": (np.asarray(reweight) - forward).tolist()})
    for row, adjusted in zip(rows, holm_adjust([row["p_value"] for row in rows])):
        row["p_value_holm"] = adjusted
    return rows
