"""Supplementary figure -- conditioning does not consistently rank subset quality.

This plot deliberately uses the frozen primary identifiability statistic: one
Spearman correlation between context condition number kappa(C) and Pdiag
generator-profile RMSE for every (N, protocol) cell.  It does not pool ranks,
does not invert kappa(C), and does not fit a pooled regression line.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common import MULTISEED, apply_style, style_axes


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "figS_conditioning_predictivity"
INPUT = MULTISEED / "identifiability" / "identifiability_correlations.csv"
PROTOCOLS = ("random", "qualified")
COLORS = {"random": "#6f6f6f", "qualified": "#4a3aa7"}
MARKERS = {"random": "o", "qualified": "s"}


def _read_rows() -> list[dict[str, object]]:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = [
        {
            "sample_size": int(row["sample_size"]),
            "protocol": row["protocol"],
            "spearman_rho": float(row["spearman_rho"]),
            "subset_count": int(row["subset_count"]),
        }
        for row in raw
        if row["metric"] == "condition_number"
        and row["recovery"] == "generator_rmse"
        and row["protocol"] in PROTOCOLS
        and row["spearman_rho"]
    ]
    expected = {(n, protocol) for n in (3, 5, 8, 10, 15, 20) for protocol in PROTOCOLS}
    observed = {(int(row["sample_size"]), str(row["protocol"])) for row in rows}
    if observed != expected or len(rows) != 12:
        raise ValueError(f"Expected 12 primary correlation cells, found {len(rows)}: {observed}")
    return sorted(rows, key=lambda row: (str(row["protocol"]), int(row["sample_size"])))


def _write_source_data(rows: list[dict[str, object]]) -> None:
    path = OUTPUT.with_name(OUTPUT.name + "_source_data.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_size", "protocol", "spearman_rho", "subset_count"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    apply_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    rows = _read_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)

    for label, protocol, ax in zip(("a", "b"), PROTOCOLS, axes):
        cell = [row for row in rows if row["protocol"] == protocol]
        ns = np.asarray([int(row["sample_size"]) for row in cell])
        rhos = np.asarray([float(row["spearman_rho"]) for row in cell])
        if np.any(ns <= 0):
            raise ValueError("Log-scale demonstration counts must be strictly positive")
        ax.axhline(0.0, color="#6f6f6f", linewidth=0.85, linestyle="--", zorder=1)
        ax.plot(
            ns,
            rhos,
            color=COLORS[protocol],
            marker=MARKERS[protocol],
            linewidth=1.55,
            markersize=4.6,
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(2.75, 22.5)
        ax.set_ylim(-1.03, 1.03)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_xlabel("demonstrations N")
        if protocol == "random":
            ax.set_ylabel("Spearman ρ: κ(C) vs generator RMSE")
        ax.set_title(f"{label}  {protocol.capitalize()} subsets", loc="left", fontsize=8.2, fontweight="bold", pad=7)
        ax.grid(axis="y", color="#d6d6d6", linestyle=(0, (1.5, 2.2)), linewidth=0.6)
        style_axes(ax)
        ax.tick_params(labelsize=7.0)
        ax.text(
            0.04,
            0.08,
            "ρ = 0: no monotonic ranking",
            transform=ax.transAxes,
            fontsize=5.75,
            color="#52514e",
        )

    fig.suptitle(
        "Simple conditioning statistics do not consistently\npredict few-shot subset quality",
        fontsize=10.8,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "Each point is the frozen within-(N, protocol) Spearman correlation over 10 subsets after averaging five execution seeds per subset.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#52514e",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.24, wspace=0.20)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    _write_source_data(rows)
    print(f"saved {OUTPUT}.{{png,pdf,svg,tiff}}")
    print("primary per-cell rho:", [(row["protocol"], row["sample_size"], row["spearman_rho"]) for row in rows])


if __name__ == "__main__":
    main()
