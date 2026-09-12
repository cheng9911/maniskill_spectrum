from __future__ import annotations

"""Shared helpers for the paper-video build: ffmpeg encode/decode, title cards,
and segment normalization + concatenation.  All rendering produces H.264
``yuv420p`` at 1920x1080 / 30 fps so the final concat is player-safe."""

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REG = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

INK = (18, 18, 18)
PAPER_WHITE = (255, 255, 255)
INDIGO = (68, 55, 168)  # our method, matches paper common.OURS


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REG
    return ImageFont.truetype(str(path), size)


def write_mp4(path: Path, frames, fps: float = FPS) -> None:
    """Encode a list of RGB uint8 frames (HxWx3) to H.264 mp4 via system ffmpeg."""
    frames = [np.ascontiguousarray(np.asarray(f)) for f in frames]
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "-", "-an",
        "-vcodec", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def read_mp4_frames(path: Path) -> list[np.ndarray]:
    """Decode an mp4 back to a list of RGB uint8 frames (HxWx3)."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {err.decode()}")
    # We do not know H/W from rawvideo alone; use ffprobe.
    w, h = _probe_size(path)
    nbytes = h * w * 3
    n = len(raw) // nbytes
    arr = np.frombuffer(raw[: n * nbytes], dtype=np.uint8).reshape(n, h, w, 3)
    return [arr[i] for i in range(n)]


def _probe_size(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
    ]
    out = subprocess.check_output(cmd).decode().strip()
    w, h = out.split(",")
    return int(w), int(h)


def title_card(lines: list[tuple[str, int, bool, tuple]], bg=(16, 16, 20)) -> np.ndarray:
    """Render a title card. ``lines`` = list of (text, font_size, bold, rgb_color)."""
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    total = sum(1.35 * size for _, size, _, _ in lines)
    y = (H - total) / 2
    for text, size, bold, color in lines:
        font = find_font(size, bold)
        box = draw.textbbox((0, 0), text, font=font)
        tw = box[2] - box[0]
        draw.text(((W - tw) / 2, y), text, font=font, fill=color)
        y += 1.35 * size
    return np.asarray(img)


def still_card(lines, seconds: float, bg=(16, 16, 20)) -> list[np.ndarray]:
    """A title card held for ``seconds`` (a list of identical frames)."""
    frame = title_card(lines, bg=bg)
    return [frame] * int(round(seconds * FPS))


def normalize_video(src: Path, dst: Path, fps: float = FPS) -> None:
    """Letterbox any source to 1920x1080@fps yuv420p for clean concatenation."""
    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps},format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-vf", vf, "-an", "-vcodec", "libx264", "-preset", "medium",
        "-crf", "20", str(dst),
    ]
    subprocess.run(cmd, check=True)


def concat_videos(paths: list[Path], out: Path) -> None:
    """Concatenate already-normalized mp4s (same codec/res/fps) into one file."""
    list_file = out.with_suffix(".txt")
    list_file.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(out),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)


def _fit_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale ``frame`` to fit inside (w, h) preserving aspect, pad with black."""
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nh, nw = int(round(fh * scale)), int(round(fw * scale))
    resized = np.asarray(Image.fromarray(frame).resize((nw, nh), Image.LANCZOS))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def tile_grid(clips: list[tuple[Path, str]], cols: int, rows: int, out: Path,
              target_frames: int = 90, gap: int = 10, title: str | None = None,
              sub: str | None = None) -> Path:
    """Tile several videos into a synchronous grid montage.

    ``clips`` = list of (source_mp4, cell_label).  Every clip is subsampled to
    ``target_frames`` and letterboxed into a cell of a ``cols`` x ``rows`` grid
    on a 1920x1080 canvas; each cell gets a small bottom label.  An optional
    title/sub caption spans the top/bottom of the whole frame.
    """
    n = len(clips)
    assert n <= cols * rows, f"{n} clips do not fit a {cols}x{rows} grid"

    # --- load every clip, resample (down- or up-) to a common length ---
    # linspace+round sub-samples longer clips and repeats shorter clips so all
    # cells stay in sync for the whole montage.
    subsampled: list[list[np.ndarray]] = []
    for path, _label in clips:
        frames = read_mp4_frames(path)
        idx = np.linspace(0, len(frames) - 1, target_frames).round().astype(int)
        subsampled.append([frames[i] for i in idx])
    t_frames = len(subsampled[0])

    # --- layout ---
    title_h = 96 if title else 0
    sub_h = 64 if sub else 0
    margin = 28
    grid_w = W - 2 * margin
    grid_h = H - title_h - sub_h - 2 * margin
    cell_w = (grid_w - (cols - 1) * gap) // cols
    cell_h = (grid_h - (rows - 1) * gap) // rows
    top = title_h + margin

    # --- static label overlay (drawn once, composited onto every frame) ---
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if title:
        tf = find_font(46, bold=True)
        tb = draw.textbbox((0, 0), title, font=tf)
        draw.text(((W - (tb[2] - tb[0])) / 2, 22), title, font=tf,
                  fill=(255, 255, 255, 255))
    if sub:
        sf = find_font(26, bold=False)
        sb = draw.textbbox((0, 0), sub, font=sf)
        draw.text(((W - (sb[2] - sb[0])) / 2, H - sub_h + 12), sub, font=sf,
                  fill=(210, 210, 210, 255))
    label_font = find_font(20, bold=True)
    for i, (_path, label) in enumerate(clips):
        r, c = divmod(i, cols)
        x0 = margin + c * (cell_w + gap)
        y0 = top + r * (cell_h + gap)
        lb = draw.textbbox((0, 0), label, font=label_font)
        lw, lh = lb[2] - lb[0], lb[3] - lb[1]
        lx, ly = x0 + 6, y0 + cell_h - lh - 10
        draw.rectangle([lx - 4, ly - 3, lx + lw + 4, ly + lh + 3],
                       fill=(0, 0, 0, 150))
        draw.text((lx, ly), label, font=label_font, fill=(255, 255, 255, 255))

    # --- compose every frame ---
    out_frames = []
    for t in range(t_frames):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        for i, (_path, _label) in enumerate(clips):
            r, c = divmod(i, cols)
            cell = _fit_frame(subsampled[i][t], cell_w, cell_h)
            x0 = margin + c * (cell_w + gap)
            y0 = top + r * (cell_h + gap)
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell
        comp = Image.alpha_composite(
            Image.fromarray(canvas).convert("RGBA"), overlay
        ).convert("RGB")
        out_frames.append(np.asarray(comp))

    write_mp4(out, out_frames, fps=FPS)
    print(f"wrote {out.name}  grid={cols}x{rows}  frames={t_frames}")
    return out
