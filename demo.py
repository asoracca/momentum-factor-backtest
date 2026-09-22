"""Offline fixture: python demo.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def timing_diagram(path):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(-12.5, 1.5)
    ax.set_ylim(-1, 1)
    ax.plot([-12, -1], [0, 0], lw=7, label="12-1 signal window")
    ax.plot([-1, 0], [0, 0], lw=7, color="gray", label="Skipped month")
    ax.plot([0, 1], [0, 0], lw=7, color="orange", label="Next-period return")
    for x, label in [(-12, "t-12"), (-1, "t-1"), (0, "t: form"), (1, "t+1: earn")]:
        ax.annotate(
            label, (x, 0), (x, 0.35), ha="center", arrowprops={"arrowstyle": "->"}
        )
    ax.axis("off")
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    from experiment import run_experiment
    from pipeline import ExperimentConfig

    root = Path(__file__).resolve().parent
    prices = pd.read_csv(root / "fixtures/prices.csv", index_col=0, parse_dates=True)
    membership = pd.read_csv(root / "fixtures/membership.csv")
    run_experiment(
        prices,
        membership,
        ExperimentConfig(
            "2023-01-31", min_assets=2, holdout_status="synthetic_validation"
        ),
        root / "outputs/demo",
        dict(
            label="SYNTHETIC point-in-time fixture; not market evidence",
            kind="synthetic",
            adjustment_policy="toy compounded levels",
            source_timestamp=None,
        ),
    )


if __name__ == "__main__":
    main()
