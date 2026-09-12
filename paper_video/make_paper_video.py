from __future__ import annotations

"""Assemble the paper demonstration video from its six segments.

Narrative order (see DESIGN.md):  why P_r(s) -> identify P_r(s) -> what P_r(s)
does -> freeze + real-robot transfer (climax, USER-SUPPLIED placeholder) ->
cross-relation breadth -> conclusion.

Segments 2/3/5 are pre-rendered by make_matrix_animation.py /
make_adapted_trajectory.py / render_libero_scene_videos.py.  This script builds
Seg 1 (opening title + keyed-insertion montage), Seg 6 (conclusion card), a Seg 4
placeholder, and concatenates everything into paper_video.mp4 + manifest.json.

Runs entirely in the ``maniskill_download`` env (PIL + matplotlib + system
ffmpeg); it only reads existing mp4s, so no SAPIEN/LIBERO dependency here.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from video_common import (
    FPS,
    H,
    W,
    concat_videos,
    find_font,
    read_mp4_frames,
    still_card,
    tile_grid,
    title_card,
    write_mp4,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEG = HERE / "segments"
KEYED_DIR = ROOT / "phase_switch_symmetry_videos"
TILT_DIR = ROOT / "real_relation_consistency" / "tilt_sweep" / "videos"

INK = (18, 18, 18)
WHITE = (255, 255, 255)
INDIGO = (68, 55, 168)
GRAY = (112, 112, 112)


# --------------------------------------------------------------------------- #
# frame helpers
# --------------------------------------------------------------------------- #
def _letterbox(frame: np.ndarray) -> np.ndarray:
    fh, fw = frame.shape[:2]
    scale = min(W / fw, H / fh)
    nh, nw = int(round(fh * scale)), int(round(fw * scale))
    resized = np.asarray(Image.fromarray(frame).resize((nw, nh), Image.LANCZOS))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    y0 = (H - nh) // 2
    x0 = (W - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _caption_frame(frame: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = find_font(42, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    sub_h = 0
    sub_font = None
    if sub:
        sub_font = find_font(28, bold=False)
        sub_h = 42
    bar_h = th + sub_h + 56
    y0 = H - bar_h - 26
    draw.rectangle([0, y0, W, H], fill=(0, 0, 0, 168))
    draw.text(((W - tw) / 2, y0 + 24), text, font=font, fill=(255, 255, 255, 255))
    if sub:
        sbbox = draw.textbbox((0, 0), sub, font=sub_font)
        sw = sbbox[2] - sbbox[0]
        draw.text(((W - sw) / 2, y0 + 24 + th + 14), sub, font=sub_font, fill=(220, 220, 220, 255))
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def _speed_to(path: Path, target_frames: int) -> list[np.ndarray]:
    """Read a source mp4 and subsample to ``target_frames`` evenly-spaced frames."""
    frames = read_mp4_frames(path)
    if len(frames) <= target_frames:
        return frames
    idx = np.linspace(0, len(frames) - 1, target_frames).round().astype(int)
    return [frames[i] for i in idx]


def caption_clip(src: Path, text: str, sub: str, target_frames: int, dst: Path) -> Path:
    frames = _speed_to(src, target_frames)
    frames = [_caption_frame(_letterbox(f), text, sub) for f in frames]
    write_mp4(dst, frames, fps=FPS)
    print(f"wrote {dst.name}  frames={len(frames)}")
    return dst


# --------------------------------------------------------------------------- #
# Seg 1 -- opening title + few-shot collection (real 3D tilt demonstrations)
# --------------------------------------------------------------------------- #
FEWSHOT_CLIPS = [
    ("pitch_plus_0deg.mp4", "demonstration 1 · pitch 0°", "N = 3 few-shot demonstrations"),
    ("pitch_plus_10deg.mp4", "demonstration 2 · pitch +10°", "N = 3 few-shot demonstrations"),
    ("pitch_minus_10deg.mp4", "demonstration 3 · pitch −10°", "N = 3 few-shot demonstrations"),
]


def build_seg1() -> Path:
    pieces: list[Path] = []

    title = title_card(
        [
            ("The relation law  P_r(s)", 72, True, WHITE),
            ("a few controlled demonstrations  →  the transferable law", 40, False, (200, 200, 210)),
        ],
        bg=(16, 16, 24),
    )
    seg1_title = SEG / "seg1_title.mp4"
    write_mp4(seg1_title, [title] * (3 * FPS), fps=FPS)
    pieces.append(seg1_title)

    for i, (fn, text, sub) in enumerate(FEWSHOT_CLIPS):
        pieces.append(caption_clip(TILT_DIR / fn, text, sub, 55, SEG / f"seg1_fewshot_{i}.mp4"))

    identify = title_card(
        [
            ("identify  P_r(s)  from these demonstrations", 56, True, INDIGO),
            ("the learned law exposes which generators matter, per phase", 34, False, (90, 90, 100)),
        ],
        bg=(245, 245, 248),
    )
    seg1_identify = SEG / "seg1_identify.mp4"
    write_mp4(seg1_identify, [identify] * (2 * FPS), fps=FPS)
    pieces.append(seg1_identify)

    out = SEG / "seg1_fewshot.mp4"
    concat_videos(pieces, out)
    print(f"wrote {out.name}  ({len(pieces)} pieces)")
    return out


# --------------------------------------------------------------------------- #
# Seg 3 -- "what P_r(s) does": 4x4 grid of SE(3) interventions + law curve
# --------------------------------------------------------------------------- #
# One cell per single-generator SE(3) intervention, rendered by
# render_se3_grid_videos.py (solve_se3 on CircularPhaseSwitchSE3-v1).  Row-major
# order matches that script's CASES.
SE3_GRID_CASES = [
    ("baseline", "baseline"),
    ("du_m15mm", "du −15 mm"),
    ("du_p15mm", "du +15 mm"),
    ("dv_m15mm", "dv −15 mm"),
    ("dv_p15mm", "dv +15 mm"),
    ("dw_m15mm", "dw −15 mm"),
    ("dw_p15mm", "dw +15 mm"),
    ("roll_m15deg", "roll −15°"),
    ("roll_p15deg", "roll +15°"),
    ("pitch_m15deg", "pitch −15°"),
    ("pitch_p15deg", "pitch +15°"),
    ("yaw_m30deg", "yaw −30°"),
    ("yaw_m15deg", "yaw −15°"),
    ("yaw_p15deg", "yaw +15°"),
    ("yaw_p30deg", "yaw +30°"),
    ("pitch_p10deg_holdout", "pitch +10° (held out)"),
]

# Second 4x4 grid: simultaneous multi-DOF interventions (coupling), matching the
# COUPLED_CASES order in render_se3_grid_videos.py.  Yaw is omitted because the
# circular hole is axisymmetric (yaw is invisible), so coupling couples the five
# meaningful generators du/dv/dw/roll/pitch.
COUPLED_GRID_CASES = [
    ("du_dv", "du + dv"),
    ("du_dw", "du + dw"),
    ("dv_dw", "dv + dw"),
    ("du_roll", "du + roll"),
    ("du_pitch", "du + pitch"),
    ("dv_roll", "dv + roll"),
    ("dv_pitch", "dv + pitch"),
    ("dw_roll", "dw + roll"),
    ("dw_pitch", "dw + pitch"),
    ("roll_pitch", "roll + pitch"),
    ("du_dv_dw", "du + dv + dw"),
    ("du_dv_roll", "du + dv + roll"),
    ("du_dv_pitch", "du + dv + pitch"),
    ("du_roll_pitch", "du + roll + pitch"),
    ("dv_roll_pitch", "dv + roll + pitch"),
    ("all5", "du+dv+dw+roll+pitch"),
]


def render_alpha_curve_card(out_path: Path, seconds: float) -> Path:
    """The identified α_pitch(s) law (real frozen-law data) as a still card."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(ROOT / "real_relation_consistency" / "tilt_sweep" / "frozen_law.npz")
    progress = data["progress"]
    phase = data["phase_codes"]
    curves = np.stack([data[f"seed_{s}_alpha_pitch"] for s in (20260910, 20270910, 20280910)])
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    phase_names = {3: "align", 4: "enter", 5: "unlock", 6: "insert"}

    bounds = []
    prev = None
    for i, p in enumerate(phase):
        if p != prev:
            bounds.append(i)
            prev = p
    bounds.append(len(phase))

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0.10, 0.22, 0.80, 0.50])
    fig.text(0.5, 0.90, "identified relation law   α_pitch(s)", ha="center", va="center",
             fontsize=46, fontweight="bold", color="#222222")
    fig.text(0.5, 0.08, "frozen from the amplitude sweep · 3 seeds", ha="center", va="center",
             fontsize=24, color="#707070")

    ax.fill_between(progress, mean - std, mean + std, color="#4437A8", alpha=0.15, linewidth=0)
    ax.plot(progress, mean, color="#4437A8", lw=3.5)
    ax.axhline(0.0, color="#cccccc", lw=1.0)
    ax.axhline(1.0, color="#cccccc", lw=1.0, ls=":")
    for j in range(len(bounds) - 1):
        x0 = bounds[j] / (len(progress) - 1)
        x1 = (bounds[j + 1] - 1) / (len(progress) - 1)
        ax.axvspan(x0, x1, color="0.90", alpha=0.4, zorder=0)
        ax.text((x0 + x1) / 2, 1.10, phase_names.get(int(phase[bounds[j]]), ""),
                ha="center", va="top", fontsize=18, color="#707070")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.24)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"], fontsize=16)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=16)
    ax.set_xlabel("phase progress  s", fontsize=20)
    ax.set_ylabel("α_pitch(s)", fontsize=24)
    ax.set_title("align 0→1   ·   enter/unlock/insert ≈1", loc="left",
                 fontsize=20, color="#4437A8", fontweight="bold", pad=8)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    write_mp4(out_path, [frame] * int(round(seconds * FPS)), fps=FPS)
    print(f"wrote {out_path.name}  frames={int(round(seconds * FPS))}")
    return out_path


def build_seg3() -> Path:
    pieces: list[Path] = []

    title = title_card(
        [
            ("How  P_r(s)  shapes the trajectory", 64, True, WHITE),
            ("the frozen law adapts to every SE(3) intervention", 38, False, (200, 200, 210)),
        ],
        bg=(16, 16, 24),
    )
    seg3_title = SEG / "seg3_title.mp4"
    write_mp4(seg3_title, [title] * (2 * FPS), fps=FPS)
    pieces.append(seg3_title)

    clips = [(SEG / f"seg3_se3_{stem}.mp4", label) for stem, label in SE3_GRID_CASES]
    grid = tile_grid(
        clips, cols=4, rows=4, out=SEG / "seg3_grid.mp4", target_frames=120,
        sub="one scene · 16 single-generator SE(3) interventions — physics-simulated rollouts",
    )
    pieces.append(grid)

    coupled_clips = [(SEG / f"seg3_coupled_{stem}.mp4", label) for stem, label in COUPLED_GRID_CASES]
    coupled_grid = tile_grid(
        coupled_clips, cols=4, rows=4, out=SEG / "seg3_coupled_grid.mp4", target_frames=120,
        title="… and to simultaneous, coupled interventions",
        sub="16 multi-DOF interventions · du/dv/dw × roll/pitch — physics-simulated rollouts",
    )
    pieces.append(coupled_grid)

    pieces.append(render_alpha_curve_card(SEG / "seg3_alpha_curve.mp4", 2.5))

    out = SEG / "seg3_multi_intervention.mp4"
    concat_videos(pieces, out)
    print(f"wrote {out.name}  ({len(pieces)} pieces)")
    return out


# --------------------------------------------------------------------------- #
# Seg 4 -- real-robot transfer (user-supplied placeholder)
# --------------------------------------------------------------------------- #
def build_seg4_placeholder() -> Path:
    card = title_card(
        [
            ("Freeze  P_sim(s)  →  real-robot transfer", 68, True, WHITE),
            ("nominal pickup  ·  tilted hole  c_10°  ·  predicted tilted response", 40, False, (200, 200, 210)),
            ("[ footage to be added — see DESIGN.md §4.3 ]", 30, False, (150, 150, 160)),
        ],
        bg=(24, 20, 16),
    )
    out = SEG / "seg4_real_transfer.mp4"
    write_mp4(out, [card] * (24 * FPS), fps=FPS)
    print(f"wrote {out.name}  frames={24 * FPS}")
    return out


# --------------------------------------------------------------------------- #
# Seg 5 -- cross-relation breadth (LIBERO 2x5 grid)
# --------------------------------------------------------------------------- #
LIBERO_CLIPS = [
    ("drawer_middle_open", "sliding  (du)"),
    ("plate_front_push", "planar push  (du)"),
    ("stove_knob_turn", "revolute knob  (yaw)"),
    ("microwave_door_revolute", "revolute door  (yaw)"),
    ("bowl_on_stove", "placement  (xyz)"),
    ("cream_cheese_in_bowl", "container-in  (xyz)"),
    ("bowl_on_plate", "stacking  (xyz)"),
    ("wine_bottle_on_rack", "rack slot  (xyz, yaw)"),
    ("wine_bottle_on_cabinet", "upright  (xyz)"),
    ("moka_pot_on_stove", "placement  (xyz)"),
]


def build_seg5() -> Path:
    clips = [(SEG / f"seg5_robot_{task_key}.mp4", label) for task_key, label in LIBERO_CLIPS]
    out = SEG / "seg5_breadth.mp4"
    tile_grid(
        clips, cols=5, rows=2, out=out, target_frames=90,
        title="The same formulation extends beyond insertion",
        sub="LIBERO · ten relations · robot-arm execution",
    )
    print(f"wrote {out.name}  ({len(clips)} scenes)")
    return out


# --------------------------------------------------------------------------- #
# Seg 6 -- conclusion
# --------------------------------------------------------------------------- #
def build_seg6() -> Path:
    card = title_card(
        [
            ("The relation law — not the trajectory —", 64, True, INDIGO),
            ("is the transferable object.", 64, True, INDIGO),
        ],
        bg=(245, 245, 248),
    )
    out = SEG / "seg6_conclusion.mp4"
    write_mp4(out, [card] * (4 * FPS), fps=FPS)
    print(f"wrote {out.name}  frames={4 * FPS}")
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_manifest(order: list[tuple[str, Path]]) -> Path:
    import json

    manifest = {
        "schema_version": 1,
        "title": "The relation law P_r(s) is the transferable object",
        "resolution": f"{W}x{H}",
        "fps": FPS,
        "segments": [
            {"id": name, "file": str(p.name), "seconds": round(_frame_count(p) / FPS, 2)}
            for name, p in order
        ],
    }
    out = SEG / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def _frame_count(path: Path) -> int:
    return len(read_mp4_frames(path))


def main() -> None:
    SEG.mkdir(parents=True, exist_ok=True)

    seg1 = build_seg1()
    seg2 = SEG / "seg2_identify_matrix.mp4"
    seg3 = build_seg3()
    seg4 = build_seg4_placeholder()
    seg5 = build_seg5()
    seg6 = build_seg6()

    order = [
        ("few_shot_collection", seg1),
        ("identify_P_r(s)", seg2),
        ("what_P_r(s)_does", seg3),
        ("real_robot_transfer", seg4),
        ("breadth_libevo", seg5),
        ("conclusion", seg6),
    ]
    manifest = build_manifest(order)
    print(f"wrote {manifest}")

    final = HERE / "paper_video.mp4"
    concat_videos([p for _, p in order], final)
    print(f"wrote {final}")


if __name__ == "__main__":
    main()
