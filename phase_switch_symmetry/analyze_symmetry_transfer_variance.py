from __future__ import annotations

"""Few-shot reliability analysis for the symmetry-intervention experiment.

The frozen benchmark already reports E_alpha (M1) as a mean over the 18
(seed x subset) cells per (task, model, N). This script asks the *second-order*
question a reviewer will ask about Pdiag finite:

    "Full operator reaches the same (or better) identification. Why Pdiag?"

The answer is not accuracy -- it is *few-shot reliability*. A flexible full
3x3 response operator can, at N=5, hallucinate a context-dependent yaw response
on a single unlucky (seed, subset) pair, whereas the generator-regularized
Pdiag finite caps the response. This script quantifies that as:

  * total std of E_alpha over the 18 cells;
  * seed-variance (std of E_alpha across execution seeds, mean over subsets);
  * subset-variance (std of E_alpha across demo subsets, mean over seeds);
  * worst-case max E_alpha;
  * hallucination rate = fraction of cells with E_alpha > 0.10 (a substantial
    spurious yaw response on the circular arm, where the oracle is 0).

The headline contrast is Full operator vs Pdiag finite on the circular arms at
N=5, where the structural prior should suppress the overfitting tail.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HALLUCINATION_THRESHOLD = 0.10
MODELS_FOCUS = ["Full operator", "Pdiag finite", "Generic RBF", "Pdiag pointwise"]
CIRCULAR_TASKS = ["circular_honest", "circular_placebo"]


def std_across_seeds(sub: pd.DataFrame) -> float:
    return float(sub.groupby("subset_id")["e_alpha"].std(ddof=0).mean())


def std_across_subsets(sub: pd.DataFrame) -> float:
    return float(sub.groupby("seed")["e_alpha"].std(ddof=0).mean())


def analyze(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task, n), sub in fits.groupby(["model", "task", "sample_size"]):
        e = sub["e_alpha"].to_numpy()
        pre = sub["alpha_pre"].to_numpy()
        rows.append(
            dict(
                model=model,
                task=task,
                sample_size=int(n),
                e_alpha_mean=float(e.mean()),
                e_alpha_total_std=float(e.std(ddof=0)),
                e_alpha_seed_std=std_across_seeds(sub),
                e_alpha_subset_std=std_across_subsets(sub),
                e_alpha_max=float(e.max()),
                e_alpha_hallucination_rate=float((e > HALLUCINATION_THRESHOLD).mean()),
                alpha_pre_mean=float(pre.mean()),
                alpha_pre_total_std=float(pre.std(ddof=0)),
                alpha_pre_max=float(pre.max()),
            )
        )
    return pd.DataFrame(rows)


def figure(variance: pd.DataFrame, output_path: Path):
    focus = variance[
        variance.model.isin(MODELS_FOCUS) & variance.task.isin(CIRCULAR_TASKS)
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    markers = {"Full operator": "o", "Pdiag finite": "s", "Generic RBF": "^",
               "Pdiag pointwise": "d"}
    for ax, (task, title) in zip(
        axes,
        [("circular_honest", "circular honest (flat yaw)"),
         ("circular_placebo", "circular placebo (uncorrelated sweep)")],
    ):
        d = focus[focus.task == task]
        for model in MODELS_FOCUS:
            row = d[d.model == model].sort_values("sample_size")
            ax.plot(row.sample_size, row.e_alpha_max, marker=markers[model],
                    label=model, linewidth=1.6)
        ax.set_xlabel("N demonstrations")
        ax.set_ylabel("worst-case E_alpha over 18 (seed, subset) cells")
        ax.set_title(title)
        ax.set_xticks([5, 10, 20])
        ax.grid(alpha=0.3)
        if task == "circular_honest":
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Few-shot reliability: worst-case generator-identification error "
                 "(lower = never hallucinates a yaw response)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fits",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/symmetry_transfer/symmetry_transfer_fits.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/symmetry_transfer/symmetry_transfer_variance.csv"
        ),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/symmetry_transfer/symmetry_transfer_variance.pdf"
        ),
    )
    args = parser.parse_args()

    fits = pd.read_csv(args.fits)
    variance = analyze(fits)
    variance.to_csv(args.output_csv, index=False)
    figure(variance, args.output_figure)

    # Headline printed table: circular arms, Full operator vs Pdiag finite.
    cols = ["model", "task", "sample_size", "e_alpha_mean", "e_alpha_total_std",
            "e_alpha_seed_std", "e_alpha_subset_std", "e_alpha_max",
            "e_alpha_hallucination_rate"]
    head = variance[
        variance.task.isin(CIRCULAR_TASKS) & variance.model.isin(MODELS_FOCUS)
    ][cols]
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("=== few-shot reliability (circular arms) ===")
    print(head.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nsaved:", args.output_csv)
    print("saved:", args.output_figure)


if __name__ == "__main__":
    main()
