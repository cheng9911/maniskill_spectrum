"""Figure 4 -- basis ablation (one-column wide).

The generator basis is principled (task-local frame), not arbitrary: conjugating
the contexts by a fixed rotation turns a cleanly diagonal response operator into
a dense one, at equal in-sample error for the full operator but a large hit for
the diagonal Pdiag model.

Two 6x6 operator heatmaps:
  (a) task-local basis  -- diagonal (off-diagonal Frobenius norm 0.15)
  (b) rotated basis     -- dense   (off-diagonal Frobenius norm 2.77)

The 6x6 matrices themselves are NOT frozen to disk (only the scalar off-diagonal
norms and e_data are), so the heatmaps are SCHEMATIC: they carry the real frozen
scalars (0.15 -> 2.77, and e_data 7.5e-6 -> 2.8e-5) as annotations and are
labelled accordingly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common import GEN_LABELS, GENS, SEQ_BLUE, apply_style, save_pub, style_axes

HERE = Path(__file__).resolve().parent

# real frozen scalars (se3_transfer_basis_ablation/basis_ablation_summary.json +
# EXPERIMENTS_RECORD.md line 665-666, 928)
OFFDIAG_LOCAL = 0.15
OFFDIAG_ROTATED = 2.77
EDATA_LOCAL = 7.5e-6
EDATA_ROTATED = 2.8e-5


def build_operators(seed: int = 0):
    """Deterministic schematic 6x6 operators with the frozen off-diag norms."""
    rng = np.random.default_rng(seed)

    def _off_diag_norm(m):
        m = np.array(m, dtype=float)
        return float(np.linalg.norm(m - np.diag(np.diag(m))))

    # local: diagonal ~1 with a tiny symmetric off-diagonal (norm 0.15)
    off = rng.standard_normal((6, 6))
    off = (off + off.T) / 2
    np.fill_diagonal(off, 0.0)
    off = off / np.linalg.norm(off) * OFFDIAG_LOCAL
    A_local = np.eye(6) + off

    # rotated: dense, diagonal ~1 with off-diagonal norm 2.77 (rotation mixes axes)
    off2 = rng.standard_normal((6, 6))
    off2 = (off2 + off2.T) / 2
    np.fill_diagonal(off2, 0.0)
    off2 = off2 / np.linalg.norm(off2) * OFFDIAG_ROTATED
    A_rot = np.eye(6) + off2

    assert abs(_off_diag_norm(A_local) - OFFDIAG_LOCAL) < 1e-9
    assert abs(_off_diag_norm(A_rot) - OFFDIAG_ROTATED) < 1e-9
    return A_local, A_rot


def panel_operator(ax, mat, title, offdiag, edata):
    ax.imshow(np.abs(mat), aspect="auto", cmap=SEQ_BLUE, vmin=0, vmax=1.2)
    ax.set_xticks(range(6))
    ax.set_xticklabels([GEN_LABELS[g] for g in GENS], fontsize=7)
    ax.set_yticks(range(6))
    ax.set_yticklabels([GEN_LABELS[g] for g in GENS], fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold")
    ax.text(0.5, -0.28, f"$\\|A_\\mathrm{{off}}\\|_F={offdiag:.2f}$   "
                       f"$e_\\mathrm{{data}}={edata:.1e}$",
            transform=ax.transAxes, ha="center", fontsize=6.5)


def main():
    apply_style()
    A_local, A_rot = build_operators()

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    panel_operator(axes[0], A_local, "a  Task-local basis",
                   OFFDIAG_LOCAL, EDATA_LOCAL)
    panel_operator(axes[1], A_rot, "b  Rotated basis",
                   OFFDIAG_ROTATED, EDATA_ROTATED)

    cbar = fig.colorbar(axes[1].images[0], ax=axes, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=6, length=0)
    cbar.set_label(r"$|A_{ij}|$  (schematic)", fontsize=6.5)

    fig.text(0.5, 0.01,
             "schematic operators; off-diagonal norms and $e_{data}$ are the "
             "frozen scalars",
             ha="center", fontsize=6, color="#555555")

    save_pub(fig, HERE / "fig4_basis_ablation")


if __name__ == "__main__":
    main()
