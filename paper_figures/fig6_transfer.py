"""Figure 6 -- zero-refit matched-relation transfer (two-column wide).

Three panels carrying the RQ4 argument (protocol -> per-target evidence ->
aggregate supervision advantage):

  (a) compact protocol strip + matched-pair table.  A source generator law
      P_a(s) identified once (N_s=30) is reused verbatim as the target law
      P_b^tr(s) = P_a(s), so the target law is fit with zero interventions
      (N_t,fit=0); the single nominal target trajectory supplies geometry only.
  (b) per-pair transferred-law error E_alpha on a log axis: every pair lands near
      the source law (1e-5 for the slide/revolute pairs, 3e-4 for the support
      pairs), far below any scratch fit.
  (c) target supervision vs. law error: the transferred law (N_t,fit=0) is lower
      than training Pdiag from scratch (N_t=3,5,8) or TP-GMM (N_t=8), with the
      generic scalar baselines ~3 orders of magnitude worse.

Data: geometry_transfer/geometry_transfer_validation.json (ours, Pdiag scratch,
frame-weighted, phase-scalar-GP) plus geometry_transfer_tpgmm_n8/
geometry_transfer_validation.json (TP-GMM SE(3) at N=8).
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import MULTISEED, save_pub, style_axes

HERE = Path(__file__).resolve().parent

OURS = "Ours transfer: source Pdiag N=30 + target nominal N=1"
SCRATCH = "Pdiag finite target scratch"
TPGMM = "TP-GMM SE(3) target scratch"
FRAME = "Frame-weighted target scratch"
PHASE = "Phase scalar GP target scratch"

PAIR_ORDER = [
    "drawer_to_plate_push",
    "knob_to_microwave_door",
    "bowl_stove_to_bowl_plate",
    "bowl_stove_to_cream_cheese_bowl",
    "bowl_stove_to_wine_cabinet",
]
PAIR_SOURCE = {
    "drawer_to_plate_push": "Drawer open",
    "knob_to_microwave_door": "Stove knob",
    "bowl_stove_to_bowl_plate": "Bowl on stove",
    "bowl_stove_to_cream_cheese_bowl": "Bowl on stove",
    "bowl_stove_to_wine_cabinet": "Bowl on stove",
}
PAIR_TARGET = {
    "drawer_to_plate_push": "Plate push",
    "knob_to_microwave_door": "Microwave door",
    "bowl_stove_to_bowl_plate": "Bowl on plate",
    "bowl_stove_to_cream_cheese_bowl": "Cream in bowl",
    "bowl_stove_to_wine_cabinet": "Wine on cabinet",
}
PAIR_RELATION = {
    "drawer_to_plate_push": r"$\delta u$: one-axis slide",
    "knob_to_microwave_door": r"$\delta\psi$: revolute",
    "bowl_stove_to_bowl_plate": "support / placement",
    "bowl_stove_to_cream_cheese_bowl": "support / container",
    "bowl_stove_to_wine_cabinet": "support / cross-object",
}

# Visual hierarchy: proposed (purple) > closest structured baseline (blue) >
# TP-GMM (gray) > generic scalar baselines (light gray).
COLOR = {
    "ours": "#4037A3",
    "pdiag_scratch": "#377EB8",
    "tpgmm": "#7F7F7F",
    "frame_scalar": "#AFAFAF",
    "phase_scalar": "#C5C5C5",
}


def _sci(x: float) -> str:
    """Compact mathtext scientific notation with round-half-up, e.g.
    2.125e-4 -> $2.13\\times10^{-4}$ (not 2.12, which is Python's half-even)."""
    d = Decimal(str(x))
    exp = d.adjusted()                       # exponent of the leading digit
    mant = d.scaleb(-exp)                    # mantissa in [1, 10)
    mant = mant.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if mant >= 10:
        mant /= 10
        exp += 1
    return rf"${mant}\times 10^{{{exp}}}$"


def load():
    main = json.loads(
        (MULTISEED / "geometry_transfer" / "geometry_transfer_validation.json").read_text()
    )
    tpgmm = json.loads(
        (MULTISEED / "geometry_transfer_tpgmm_n8" / "geometry_transfer_validation.json").read_text()
    )
    return main["pairs"], main["by_pair"], main["summary"], tpgmm["summary"]


def _val(rows, method: str, n: int) -> float:
    for r in rows:
        if r["method"] == method and r["sample_size"] == n:
            return float(r["e_alpha_mean"])
    raise ValueError(f"missing {method!r} @ N={n}")


def _pair_val(by_pair, method: str, pair_key: str, n: int) -> float:
    for r in by_pair:
        if r["method"] == method and r["pair_key"] == pair_key and r["sample_size"] == n:
            return float(r["e_alpha_mean"])
    raise ValueError(f"missing {method!r} {pair_key!r} @ N={n}")


def panel_transfer(ax):
    """(a) compact protocol strip + matched-pair table (no flowchart boxes)."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(a)  Zero-refit matched-relation transfer", loc="left",
                 fontsize=8.2, fontweight="semibold", pad=5)

    # ---- protocol strip ----
    ax.text(0.06, 0.84, r"source: $N_s=30$ interventions",
            ha="left", va="center", fontsize=6.3, color="0.30")
    ax.text(0.24, 0.76, r"identify $P_a(s)$",
            ha="center", va="center", fontsize=6.0, color="0.45")
    ax.annotate("", xy=(0.69, 0.84), xytext=(0.31, 0.84),
                arrowprops=dict(arrowstyle="->", lw=1.25, color=COLOR["ours"]))
    ax.text(0.50, 0.875, "reuse source law",
            ha="center", va="bottom", fontsize=6.1, color=COLOR["ours"])
    ax.text(0.50, 0.78, r"$P_b^{\mathrm{tr}}(s)=P_a(s),\qquad N_{t,\mathrm{fit}}=0$",
            ha="center", va="center", fontsize=7.3, color=COLOR["ours"], fontweight="semibold")
    ax.text(0.94, 0.84, "target: 1 nominal trajectory",
            ha="right", va="center", fontsize=6.3, color="0.30")
    ax.text(0.94, 0.76, "geometry only",
            ha="right", va="center", fontsize=6.0, color="0.45", fontstyle="italic")

    ax.axhline(0.67, color="0.84", lw=0.6)

    # ---- matched-pair table ----
    x_src, x_rel, x_tgt = 0.04, 0.50, 0.96
    ax.text(x_src, 0.59, "source task", ha="left", fontsize=6.1,
            color="0.42", fontweight="semibold")
    ax.text(x_rel, 0.59, "matched generator semantics", ha="center", fontsize=6.1,
            color="0.42", fontweight="semibold")
    ax.text(x_tgt, 0.59, "target task", ha="right", fontsize=6.1,
            color="0.42", fontweight="semibold")

    ys = np.linspace(0.49, 0.09, len(PAIR_ORDER))
    for pk, y in zip(PAIR_ORDER, ys):
        ax.text(x_src, y, PAIR_SOURCE[pk], ha="left", va="center",
                fontsize=6.4, color="0.12")
        ax.text(x_rel, y, PAIR_RELATION[pk], ha="center", va="center",
                fontsize=6.2, color="0.35")
        ax.text(x_tgt, y, PAIR_TARGET[pk], ha="right", va="center",
                fontsize=6.4, color="0.12")


def panel_per_pair(ax, by_pair):
    """(b) horizontal dot plot of the transferred-law error per pair (log x)."""
    vals = [_pair_val(by_pair, OURS, pk, 8) for pk in PAIR_ORDER]
    labels = [PAIR_TARGET[pk] for pk in PAIR_ORDER]
    y = np.arange(len(vals))[::-1]

    ax.scatter(vals, y, s=30, color=COLOR["ours"], zorder=3)
    for v, yi in zip(vals, y):
        ax.text(v * 1.3, yi, _sci(v), va="center", ha="left",
                fontsize=5.8, color="0.25")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(8e-6, 9e-4)
    ax.set_xlabel(r"transferred-law error $E_\alpha$", fontsize=7)
    ax.set_title("(b)  Per-pair transferred-law error", loc="left",
                 fontsize=8.5, fontweight="semibold", pad=6)
    ax.grid(axis="x", color="0.88", lw=0.5, ls=(0, (2, 2)))
    style_axes(ax)
    ax.tick_params(labelsize=6.5)


def panel_supervision(ax, summary, tpgmm_summary):
    """(c) supervision vs. law error: ours sits at N_t,fit = 0."""
    ours = _val(summary, OURS, 8)
    ns = [3, 5, 8]
    pdiag = [_val(summary, SCRATCH, n) for n in ns]
    tpgmm = _val(tpgmm_summary, TPGMM, 8)
    scalar = _val(summary, FRAME, 8)  # frame-weighted ~ phase-scalar (~0.127)

    ax.scatter([0], [ours], s=48, color=COLOR["ours"], zorder=5, edgecolors="none")
    ax.plot(ns, pdiag, color=COLOR["pdiag_scratch"], lw=1.5, marker="o", ms=4, zorder=4)
    ax.scatter([8], [tpgmm], s=34, color=COLOR["tpgmm"], zorder=4,
               marker="^", edgecolors="none")
    ax.scatter([8], [scalar], s=26, color=COLOR["frame_scalar"], zorder=3,
               marker="s", edgecolors="none")

    # direct labels (no legend needed)
    ax.text(0.28, ours * 1.5, r"Ours: $N_{t,\mathrm{fit}}=0$",
            fontsize=6.3, color=COLOR["ours"], ha="left", va="center")
    ax.text(0.28, ours * 0.55, _sci(ours),
            fontsize=5.8, color=COLOR["ours"], ha="left", va="center")
    ax.text(8.25, pdiag[-1] * 1.25, "Pdiag scratch",
            fontsize=6.0, color=COLOR["pdiag_scratch"], ha="left", va="center")
    ax.text(8.25, tpgmm * 0.8, "TP-GMM",
            fontsize=6.0, color=COLOR["tpgmm"], ha="left", va="center")
    ax.text(8.25, scalar * 1.12, "scalar baselines",
            fontsize=5.8, color="0.45", ha="left", va="center")

    ax.set_yscale("log")
    ax.set_ylim(8e-5, 3e-1)
    ax.set_xlim(-0.9, 11.6)
    ax.set_xticks([0, 3, 5, 8])
    ax.set_xlabel(r"target interventions used for law fitting $N_{t,\mathrm{fit}}$", fontsize=7)
    ax.set_ylabel(r"$E_\alpha$", fontsize=7)
    ax.set_title("(c)  Target supervision vs. law error", loc="left",
                 fontsize=8.5, fontweight="semibold", pad=6)
    ax.text(0.015, 0.94, "lower is better", transform=ax.transAxes,
            fontsize=5.6, color="0.5", ha="left", va="top", fontstyle="italic")
    ax.grid(axis="y", color="0.88", lw=0.5, ls=(0, (2, 2)))
    style_axes(ax)
    ax.tick_params(labelsize=6.5)


def main():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.5,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    pairs, by_pair, summary, tpgmm_summary = load()
    if [p["pair_key"] for p in pairs] != PAIR_ORDER:
        raise ValueError("pair order in JSON does not match PAIR_ORDER")

    fig = plt.figure(figsize=(7.1, 4.45))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.62, 1.0],
                  hspace=0.55, wspace=0.36, left=0.09, right=0.97,
                  top=0.94, bottom=0.11)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_transfer(ax_a)
    panel_per_pair(ax_b, by_pair)
    panel_supervision(ax_c, summary, tpgmm_summary)

    save_pub(fig, HERE / "fig6_transfer")

    print("panel (b) per-pair transferred-law error (ours, N=8):")
    for pk in PAIR_ORDER:
        print(f"  {pk:35s} {_pair_val(by_pair, OURS, pk, 8):.3e}")
    print("panel (c) supervision:")
    print(f"  ours (N_t,fit=0): {_val(summary, OURS, 8):.3e}")
    for n in (3, 5, 8):
        print(f"  pdiag scratch N={n}: {_val(summary, SCRATCH, n):.3e}")
    print(f"  tpgmm N=8: {_val(tpgmm_summary, TPGMM, 8):.3e}")
    print(f"  scalar baselines (frame ~ phase): {_val(summary, FRAME, 8):.3e} / "
          f"{_val(summary, PHASE, 8):.3e}")


if __name__ == "__main__":
    main()
