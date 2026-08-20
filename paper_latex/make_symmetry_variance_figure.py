from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "phase_switch_symmetry_multiseed" / "symmetry_transfer" / "symmetry_transfer_variance.csv"

MODELS = ["Full operator", "Generic RBF", "Pdiag finite"]
DISPLAY = {
    "Full operator": "Full operator",
    "Generic RBF": "Generic RBF",
    "Pdiag finite": "Pdiag finite",
}
COLORS = {
    "Full operator": "#6f6f6f",
    "Generic RBF": "#2c7c59",
    "Pdiag finite": "#1f77b4",
}
MARKERS = {
    "Full operator": "o",
    "Generic RBF": "s",
    "Pdiag finite": "^",
}


def load_rows(task: str = "circular_honest") -> dict[str, list[dict[str, float]]]:
    rows: dict[str, list[dict[str, float]]] = {model: [] for model in MODELS}
    with DATA.open(newline="") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            if row["task"] != task or model not in rows:
                continue
            rows[model].append(
                {
                    "N": float(row["sample_size"]),
                    "std": float(row["e_alpha_total_std"]),
                    "worst": float(row["e_alpha_max"]),
                }
            )
    for model_rows in rows.values():
        model_rows.sort(key=lambda item: item["N"])
    return rows


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["1e-4", "1e-3", "1e-2", "1e-1"])
    ax.set_xlabel("Demonstrations, N")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=7.0, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    rows = load_rows()
    assert all(row["std"] > 0 and row["worst"] > 0 for model_rows in rows.values() for row in model_rows)
    fig, ax = plt.subplots(1, 1, figsize=(3.35, 1.85))

    for model in MODELS:
        xs = [row["N"] for row in rows[model]]
        ys = [row["std"] for row in rows[model]]
        ax.plot(
            xs,
            ys,
            marker=MARKERS[model],
            markersize=4.2,
            linewidth=1.45,
            color=COLORS[model],
            label=DISPLAY[model],
        )
        ax.text(
            xs[-1] + 0.55,
            ys[-1],
            DISPLAY[model],
            ha="left",
            va="center",
            fontsize=6.8,
            color=COLORS[model],
        )

    style_axis(ax, "Std. of E-alpha")
    ax.set_title(
        "Few-shot law stability on circular-honest arm",
        fontsize=7.6,
        fontweight="bold",
        pad=3,
    )
    ax.set_xlim(4.4, 27.0)
    ax.set_ylim(4e-5, 4e-1)
    fig.tight_layout(pad=0.55)

    base = HERE / "symmetry_variance"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
