from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FRAME_DIR = ROOT / "phase_switch_symmetry_rollouts_rotated" / "_scene_frames"

C_DARK = "#1b4f72"
C_MID = "#2471a3"
C_AMBER = "#b9770e"


PANELS = [
    ("keyedQ1_align_keyed.png", "Align keyed", C_MID),
    ("keyedQ1_enter_key.png", "Enter key", C_MID),
    ("keyedQ1_unlock_yaw.png", "Unlock yaw", C_MID),
    ("keyedQ1_circular_insert.png", "Circular insert", C_MID),
    ("keyedQ1_align_keyed.png", "Keyed Q1 vertical", C_MID),
    ("keyedQ2_circular_insert.png", "Keyed Q2 horizontal", C_AMBER),
    ("honest_circular_insert.png", "Circular honest", C_AMBER),
    ("placebo_circular_insert.png", "Circular placebo", C_AMBER),
]


def load_image(name: str) -> Image.Image:
    path = FRAME_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
        }
    )

    fig = plt.figure(figsize=(7.2, 3.55), facecolor="white")
    gs = fig.add_gridspec(
        2,
        4,
        left=0.025,
        right=0.99,
        top=0.84,
        bottom=0.075,
        wspace=0.06,
        hspace=0.18,
    )

    for index, (filename, title, edge) in enumerate(PANELS):
        ax = fig.add_subplot(gs[index // 4, index % 4])
        ax.imshow(load_image(filename))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(edge)
            spine.set_linewidth(2.2)
        ax.set_title(title, fontsize=6.8, fontweight="bold", color=C_DARK, pad=4)

    fig.text(
        0.5,
        0.94,
        "Keyed-to-circular peg insertion",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=C_DARK,
    )
    fig.text(
        0.006,
        0.60,
        "Phases",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=C_MID,
        rotation_mode="anchor",
    )
    fig.text(
        0.006,
        0.265,
        "Variants",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=C_AMBER,
        rotation_mode="anchor",
    )

    outputs = [
        ROOT / "experiment_overview_scenes.png",
        ROOT / "experiment_overview_scenes.pdf",
        ROOT / "experiment_overview_scenes.svg",
        ROOT / "experiment_overview_scenes.tiff",
        HERE / "experiment_overview_scenes.png",
        HERE / "experiment_overview_scenes.pdf",
        HERE / "experiment_overview_scenes.svg",
        HERE / "experiment_overview_scenes.tiff",
    ]
    for out in outputs:
        dpi = 600 if out.suffix in {".png", ".tiff"} else None
        fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
