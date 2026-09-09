"""Shared styling + data helpers for the paper_figures scripts.

Paper-wide color semantics (Nature-like fixed palette, light mode):

    indigo  OURS    #4437A8   our method (Pdiag finite)
    blue    COMP    #3278B8   comparison baseline (TP-GMM SE(2))
    vermilion CONTROL #E66532 control / gauge condition
    grays   GRAY_*           neutral ink / grid / muted text

These slots are fixed and non-negotiable: a color carries the same meaning in
every figure.  Relevance magnitude uses a single white -> indigo sequential ramp
(``RELEVANCE_CMAP``), shared by the Fig. 3 bubble matrix and heatmap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MULTISEED = ROOT / "phase_switch_symmetry_multiseed"
ROLLOUTS = ROOT / "phase_switch_symmetry_rollouts_rotated"

# --------------------------------------------------------------------------
# Paper-wide color semantics (Nature-like fixed palette)
# --------------------------------------------------------------------------
# Fixed, non-negotiable slots.  Indigo = our method; blue = comparison baseline;
# vermilion = control/gauge condition; grays = neutral ink/grid.  Do not reuse a
# slot for a different role within a single figure.
OURS = "#4437A8"         # indigo    -- our method (Pdiag finite)
COMP = "#3278B8"         # blue      -- comparison baseline (TP-GMM SE(2))
CONTROL = "#E66532"      # vermilion -- control / gauge condition
GRAY_DARK = "#707070"
GRAY_MID = "#A0A0A0"
GRAY_LIGHT = "#C7C7C7"
GRID = "#DDDDDD"
TEXT = "#222222"

# 3 conditions (fixed semantic mapping: keyed=active, honest/placebo=gauge control)
CONDITION_COLORS = {
    "keyed": "#3278B8",            # blue      (active condition)
    "circular_honest": "#E66532",  # vermilion (gauge control)
    "circular_placebo": "#A0A0A0", # gray      (placebo control)
}
CONDITION_LABELS = {
    "keyed": "Keyed",
    "circular_honest": "Circular honest",
    "circular_placebo": "Circular placebo",
}

# 4 models (hero = ours in indigo, then fixed comparison order)
MODEL_COLORS = {
    "Pdiag finite": "#4437A8",     # indigo    (ours)
    "TP-GMM SE(2)": "#3278B8",     # blue      (comparison)
    "Full operator": "#E66532",    # vermilion (comparison)
    "Generic RBF": "#707070",      # gray      (comparison)
}
MODEL_LINESTYLES = {
    "Pdiag finite": "-",
    "TP-GMM SE(2)": "-.",
    "Full operator": "--",
    "Generic RBF": ":",
}
MODEL_MARKERS = {
    "Pdiag finite": "o",
    "TP-GMM SE(2)": "^",
    "Full operator": "s",
    "Generic RBF": "D",
}
MODEL_LABELS = {
    "Pdiag finite": "Pdiag (ours)",
    "TP-GMM SE(2)": "TP-GMM",
    "Full operator": "Full",
    "Generic RBF": "RBF",
}

# Relevance magnitude ramp: white -> indigo (single hue, light -> dark).  Both the
# Fig. 3 bubble matrix and the multi-generator heatmap encode the same quantity
# (mean learned relevance alpha), so they deliberately share this one ramp.
_RELEVANCE = ["#F7F7FA", "#E9E8F3", "#D7D4EA", "#BBB6DE", "#9890CF", "#7166BC", "#4437A8"]
RELEVANCE_CMAP = mpl.colors.LinearSegmentedColormap.from_list("relevance", _RELEVANCE, N=256)

# Sequential blue ramp (kept for backward compatibility with older figures)
_SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
]
SEQ_BLUE = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE, N=256)

NEUTRAL_GRAY = "#9aa0a6"
INK = TEXT
MUTED = "#52514e"


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "axes.edgecolor": GRID,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
    })


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7, width=0.7, length=3)


def save_pub(fig, path: Path, dpi: int = 600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    print(f"saved {path}.{{png,pdf,svg}}")


# --------------------------------------------------------------------------
# SE(3) generator conventions
# --------------------------------------------------------------------------
GENS = ("du", "dv", "dw", "roll", "pitch", "yaw")
GEN_LABELS = {
    "du": "d$u$",
    "dv": "d$v$",
    "dw": "d$w$",
    "roll": "d$\\phi$",
    "pitch": "d$\\theta$",
    "yaw": "d$\\psi$",
}
PHASE_NAMES = ("Align", "Enter", "Unlock", "Insert")
PHASE_BOUNDS = (0.25, 0.5, 0.75)

# arm -> H5 rollout file (seed 20260818)
ARM_H5 = {
    "keyed": "rotated_Q1_seed_20260818.h5",
    "circular_honest": "circular_honest_seed_20260818.h5",
    "circular_placebo": "circular_placebo_seed_20260818.h5",
}


def _import_benchmark():
    import sys
    sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
    from analyze_phase_switch_rollouts import episode_keys  # noqa: F401
    from benchmark_phase_switch_baselines import (  # noqa: F401
        empirical_isolated_profile,
        progress_grid,
        task_curve,
        usable,
    )
    return sys.modules["benchmark_phase_switch_baselines"]


def empirical_yaw_response(arm: str, bins: int = 25):
    """Return (progress, phase_codes, g_yaw, nominal_yaw_deg) for one arm.

    ``g_yaw`` is the empirical yaw intervention response g_psi(s) = d(delta_psi)/
    d(Delta_psi), the third column of ``empirical_isolated_profile``.
    ``nominal_yaw_deg`` is the baseline (zero-intervention) episode's unwrapped
    peg yaw in degrees over progress.
    """
    import h5py
    bm = _import_benchmark()
    progress, phase_codes = bm.progress_grid(bins)
    with h5py.File(ROLLOUTS / ARM_H5[arm], "r") as f:
        keys = bm.episode_keys(f)
        usable_keys = [k for k in keys if bm.usable(f[k])]
        isolated_all = [
            k for k in usable_keys
            if str(f[k].attrs.get("generator", "")) in {"yaw", "translation"}
        ]
        baseline = [
            k for k in isolated_all
            if np.linalg.norm(np.asarray(f[k]["causal_delta"])) < 1e-12
        ][0]
        iso = [k for k in isolated_all if k != baseline]
        nominal = bm.task_curve(f[baseline], bins)[:, 2]
        prof = bm.empirical_isolated_profile(f, baseline, iso, bins)
    return progress, phase_codes, prof[:, 2], np.rad2deg(nominal)
