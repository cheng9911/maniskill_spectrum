from __future__ import annotations

"""Segment 2 -- "Identify P_r(s)": animate the generator-relevance bubble matrix.

Reveals the 6-generator x 13-task structure, then walks the four relation
families (translation -> revolute -> placement -> insertion) so the matrix reads
as "the learned law exposes which task-local generators matter", not a data dump.

Reuses the frozen matrix data + styling from paper_figures/fig3_se3_generators.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from video_common import W, H, FPS, write_mp4  # noqa: E402  (paper_video/video_common.py)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paper_figures"))

import common as pcommon  # noqa: E402  (paper_figures/common.py)
import fig3_se3_generators as fig3  # noqa: E402

# (family name, start_col, end_col, description) -- matches fig3.FAMILY_LABELS.
FAMILIES = [
    ("Translation / planar", 0, 4, "du, dv relevance in sliding & pushing"),
    ("Revolute", 4, 6, "yaw relevance in knobs & hinged doors"),
    ("Placement", 6, 12, "upright xyz placement across supports"),
    ("Insertion", 12, 13, "keyed insertion, phase-structured relevance"),
]

GEN_LABELS = [pcommon.GEN_LABELS[g] for g in pcommon.GENS]
TASK_NAMES = [t[0] for t in fig3.TASKS]

T_REVEAL0, T_REVEAL1 = 60, 180
T_FAM = 180
FAM_LEN = 60
TOTAL = 480


def draw(frame_idx: int, mat, ax):
    ax.clear()
    n_tasks = len(fig3.TASKS)
    n_gens = 6

    reveal = 0.0
    if frame_idx >= T_REVEAL1:
        reveal = 1.0
    elif frame_idx >= T_REVEAL0:
        reveal = (frame_idx - T_REVEAL0) / (T_REVEAL1 - T_REVEAL0)

    highlight = -1
    if frame_idx >= T_FAM:
        fam = (frame_idx - T_FAM) // FAM_LEN
        if fam < len(FAMILIES):
            highlight = fam

    norm = plt.Normalize(0.0, fig3.ALPHA_MAX)
    for gi in range(n_gens):
        for ti in range(n_tasks):
            if ti > reveal * (n_tasks - 1) + 1e-9:
                continue
            value = float(mat[ti, gi])
            if value < fig3.DOT_THRESHOLD:
                continue
            if highlight >= 0:
                _, s, e, _ = FAMILIES[highlight]
                alpha = 1.0 if s <= ti < e else 0.12
            else:
                alpha = 1.0
            color = fig3.STRUCT_CMAP(norm(value))
            ax.scatter([ti], [gi], s=fig3.BUBBLE_AREA_SCALE * value,
                       color=color, edgecolors="none", alpha=alpha, zorder=3)

    for x in fig3.FAMILY_DIVIDERS:
        ax.axvline(x, color=pcommon.GRID, lw=1.0, zorder=1)

    trans = ax.get_xaxis_transform()
    for name, s, e, _ in FAMILIES:
        center = (s + e - 1) / 2
        active = highlight >= 0 and FAMILIES[highlight][0] == name
        ax.text(center, 1.02, name, transform=trans, ha="center", va="bottom",
                fontsize=15, color=pcommon.GRAY_DARK,
                fontweight="bold" if active else "normal")

    if highlight >= 0:
        _, s, e, _ = FAMILIES[highlight]
        rect = plt.Rectangle((s - 0.5, -0.6), e - s, 6.2, fill=False,
                             edgecolor=pcommon.OURS, lw=3.0, zorder=4)
        ax.add_patch(rect)

    ax.set_xlim(-0.6, n_tasks - 0.4)
    ax.set_ylim(-0.6, 5.6)
    ax.invert_yaxis()
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(TASK_NAMES, fontsize=11, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(n_gens))
    ax.set_yticklabels(GEN_LABELS, fontsize=16)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    out_dir = HERE / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "seg2_identify_matrix.mp4"

    mat = fig3.mean_alpha_matrix()
    assert mat.shape == (len(fig3.TASKS), 6), mat.shape

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0.07, 0.24, 0.86, 0.58])
    fig.text(0.5, 0.945, "Identify the relation law  P$_r$(s)",
             ha="center", va="center", fontsize=40, fontweight="bold",
             color=pcommon.INK)
    cap_text = fig.text(0.5, 0.13, "", ha="center", va="center",
                        fontsize=24, color=pcommon.INK)
    cap_sub = fig.text(0.5, 0.065, "", ha="center", va="center",
                       fontsize=17, color=pcommon.GRAY_DARK)

    frames = []
    for t in range(TOTAL):
        if t < T_REVEAL0:
            cap = "controlled interventions  {c$^{(n)}$, X$^{(n)}$(s)}  →  P$_r$(s)"
            sub = "few-shot: N = 3 / 5 / 8 demonstrations"
        elif t < T_REVEAL1:
            cap = "The learned law exposes which task-local generators matter"
            sub = "across different relations"
        else:
            fam = min((t - T_FAM) // FAM_LEN, len(FAMILIES) - 1)
            name, _, _, desc = FAMILIES[fam]
            cap = f"{name}: {desc}"
            sub = "α$_j$(s) — per-phase generator relevance"
        cap_text.set_text(cap)
        cap_sub.set_text(sub)

        draw(t, mat, ax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    write_mp4(out, frames, fps=FPS)
    print(f"wrote {out}  frames={len(frames)}")


if __name__ == "__main__":
    main()
