
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Paths
RESULTS_DIR = Path(__file__).resolve().parent / "evaluation_results"
INPUT_FILE = RESULTS_DIR / "accuracy_per_run.csv"

OUTPUT_CSV = RESULTS_DIR / "paired_comparison.csv"
OUTPUT_FIGURE = RESULTS_DIR / "paired_accuracy.png"

METHODS = ["Forward", "Reweight-R"]
DATASETS = ["FashionMNIST0.3", "FashionMNIST0.6", "CIFAR"]


def compare_runs():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Run summarize_evaluation.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "dataset", "seed", "method", "accuracy"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError("Missing required columns in results CSV.")

    comparisons = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, dataset in zip(axes, DATASETS):
        data = df[df["dataset"] == dataset]

        if data.empty:
            raise ValueError(f"No results for {dataset}")

        # Pair both methods using the same random seed
        paired = data.pivot(
            index="seed",
            columns="method",
            values="accuracy"
        ).sort_index()

        if not set(METHODS).issubset(paired.columns):
            raise ValueError(f"Missing classifier results: {dataset}")

        paired = paired[METHODS].dropna()

        if len(paired) < 10:
            raise ValueError(
                f"{dataset}: expected at least 10 paired runs."
            )

        forward = paired["Forward"].to_numpy()
        reweight = paired["Reweight-R"].to_numpy()
        seeds = paired.index.to_numpy()

        # Calculate paired accuracy differences
        differences = reweight - forward

        comparison = {
            "dataset": dataset,
            "n_runs": len(paired),
            "forward_mean": np.mean(forward),
            "forward_std": np.std(forward, ddof=0),
            "reweight_mean": np.mean(reweight),
            "reweight_std": np.std(reweight, ddof=0),
            "mean_difference": np.mean(differences),
            "difference_std": np.std(differences, ddof=0),
            "forward_wins": int(np.sum(differences < 0)),
            "reweight_wins": int(np.sum(differences > 0)),
            "ties": int(np.sum(differences == 0)),
        }

        comparisons.append(comparison)

        # Plot accuracy for each random seed
        ax.plot(
            seeds,
            forward,
            marker="o",
            label="Forward Correction",
        )

        ax.plot(
            seeds,
            reweight,
            marker="s",
            label="Reweight-R",
        )

        ax.set_title(dataset)
        ax.set_xlabel("Random seed")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_xticks(seeds)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Save the graph
    fig.suptitle(
        "Paired Classification Accuracy Across Random Seeds",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save the comparison table
    results = pd.DataFrame(comparisons)
    results.to_csv(OUTPUT_CSV, index=False)

    print("\nPAIRED CLASSIFIER COMPARISON")
    print("=" * 65)

    for row in comparisons:
        print(f"\nDataset: {row['dataset']}")
        print(
            f"Forward:    {row['forward_mean']:.2f} "
            f"+/- {row['forward_std']:.2f}%"
        )
        print(
            f"Reweight-R: {row['reweight_mean']:.2f} "
            f"+/- {row['reweight_std']:.2f}%"
        )
        print(
            f"Mean difference (Reweight - Forward): "
            f"{row['mean_difference']:+.2f} pp"
        )
        print(f"Forward wins: {row['forward_wins']}")
        print(f"Reweight wins: {row['reweight_wins']}")
        print(f"Ties: {row['ties']}")

    print("\nSaved outputs:")
    print(OUTPUT_CSV)
    print(OUTPUT_FIGURE)


if __name__ == "__main__":
    compare_runs()
