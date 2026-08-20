from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "phase_switch_symmetry_multiseed" / "symmetry_transfer"

MODELS = [
    "Frame-weighted",
    "TP-GMM SE(2)",
    "Generic RBF",
    "Full operator",
    "Pdiag finite",
]

MODEL_LABELS = {
    "Frame-weighted": "Frame\nweighted",
    "TP-GMM SE(2)": "TP-GMM\nSE(2)",
    "Generic RBF": "Generic\nRBF",
    "Full operator": "Full\noperator",
    "Pdiag finite": "Pdiag\nfinite",
}

TASK_LABELS = {
    "keyed": "Keyed",
    "circular_honest": "Circular\nhonest",
    "circular_placebo": "Circular\nplacebo",
}

COLORS = {
    "Frame-weighted": "#d28b00",
    "TP-GMM SE(2)": "#168aad",
    "Generic RBF": "#7f7f7f",
    "Full operator": "#b8b8b8",
    "Pdiag finite": "#111111",
    "keyed": "#315f9f",
    "circular_honest": "#1f9d55",
    "circular_placebo": "#b26a00",
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def get_float(rows: list[dict[str, str]], **query: object) -> float:
    for row in rows:
        ok = True
        for key, value in query.items():
            if str(row[key]) != str(value):
                ok = False
                break
        if ok:
            for field in ("mean", "accuracy", "G_psi_mean"):
                if field in row:
                    return float(row[field])
    raise KeyError(query)


def get_std(rows: list[dict[str, str]], **query: object) -> float:
    for row in rows:
        if all(str(row[key]) == str(value) for key, value in query.items()):
            for field in ("std", "G_psi_std"):
                if field in row:
                    return float(row[field])
    return 0.0


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, width=0.7, length=3)


def panel_design(ax):
    ax.axis("off")
    ax.set_title("A  Three matched intervention arms", loc="left", fontsize=8.5, fontweight="bold")

    xs = [0.035, 0.365, 0.695]
    titles = ["Keyed", "Circular honest", "Circular placebo"]
    desc = [
        "Delta psi changes the gate\nalpha yaw high before clearance",
        "Delta psi is gauge\nalpha yaw remains zero",
        "15 deg yaw sweep present\nbut fixed across Delta psi",
    ]
    edge = [COLORS["keyed"], COLORS["circular_honest"], COLORS["circular_placebo"]]
    for x, title, text, color in zip(xs, titles, desc, edge):
        patch = FancyBboxPatch(
            (x, 0.22),
            0.27,
            0.55,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#f8f8f8",
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + 0.135, 0.66, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + 0.135, 0.43, text, ha="center", va="center", fontsize=6.8, linespacing=1.25)

    ax.annotate(
        "same contexts, seeds, controller, anchor and retry budget",
        xy=(0.5, 0.1),
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
        xycoords="axes fraction",
    )


def panel_m1(ax, rows):
    tasks = ["keyed", "circular_honest", "circular_placebo"]
    values = [
        [get_float(rows, task=task, model=model, sample_size=20) for task in tasks]
        for model in MODELS
    ]
    im = ax.imshow(values, aspect="auto", cmap="viridis_r", vmin=0, vmax=0.36)
    for i, row in enumerate(values):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.7, color="white" if val > 0.16 else "black")
    ax.set_title("B  Generator-law error at N=20", loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=6.1)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=6.1)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_m2(ax, rows):
    ns = [5, 10, 20]
    for model in MODELS:
        vals = [get_float(rows, model=model, sample_size=n) for n in ns]
        ax.plot(
            ns,
            vals,
            marker="o",
            linewidth=1.6 if model == "Pdiag finite" else 1.2,
            color=COLORS[model],
            label=MODEL_LABELS[model].replace("\n", " "),
        )
    ax.axhline(1.0, color="#cfcfcf", linewidth=0.8, zorder=0)
    ax.set_title("C  Keyed-vs-circular discrimination", loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Mixed demonstrations", fontsize=7)
    ax.set_ylabel("M2 accuracy", fontsize=7)
    ax.set_xticks(ns)
    ax.set_ylim(0.65, 1.03)
    style_axes(ax)


def panel_gap(ax, rows):
    ns = [5, 10, 20]
    for model in MODELS:
        vals = [
            get_float(rows, model=model, sample_size=n, arm="circular_placebo")
            for n in ns
        ]
        errs = [
            get_std(rows, model=model, sample_size=n, arm="circular_placebo")
            for n in ns
        ]
        ax.errorbar(
            ns,
            vals,
            yerr=errs,
            marker="o",
            capsize=2,
            linewidth=1.6 if model == "Pdiag finite" else 1.2,
            color=COLORS[model],
            label=MODEL_LABELS[model].replace("\n", " "),
        )
    ax.axhline(1.0, color="#d0d0d0", linestyle="--", linewidth=0.9)
    ax.set_title("D  Placebo symmetry gap", loc="left", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Mixed demonstrations", fontsize=7)
    ax.set_ylabel("Gpsi (ideal = 1)", fontsize=7)
    ax.set_xticks(ns)
    ax.set_ylim(0, 1.08)
    style_axes(ax)


def main():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    m1 = read_rows("symmetry_transfer_M1_ealpha.csv")
    m2 = read_rows("symmetry_transfer_M2_discrimination.csv")
    gap = read_rows("symmetry_transfer_M2b_symmetry_gap.csv")

    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.72, 0.09, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_leg = fig.add_subplot(gs[1, :])
    ax_b = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[2, 1])
    ax_d = fig.add_subplot(gs[2, 2])

    panel_design(ax_a)
    panel_m1(ax_b, m1)
    panel_m2(ax_c, m2)
    panel_gap(ax_d, gap)
    handles, labels = ax_c.get_legend_handles_labels()
    ax_leg.axis("off")
    ax_leg.legend(handles, labels, ncol=5, loc="center", fontsize=6.1, handlelength=1.6)

    out = HERE / "symmetry_transfer_main"
    fig.savefig(out.with_suffix(".png"), dpi=500, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
