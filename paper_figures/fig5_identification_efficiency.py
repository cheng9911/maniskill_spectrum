"""Figure 5 -- identification efficiency under limited interventions.

Two panels answering the core RQ3 question (does the structured Pdiag prior
recover the correct law, and reproduce the task, from few interventions):

  (a) generator-law error E_alpha versus the number of matched demonstrations;
  (b) held-out task reproduction error E_task versus the same demonstrations.

Both panels use every successful matched seed-subset fit shared by the four
models at N=5,10,20,30 (matched cells do not exist at N=3).  Points are
arithmetic means and error bars are descriptive s.e.m. over those fixed
seed-subset cells. The complete-law recovery diagnostic is computed and written
to the source-data CSV but not plotted: at N=3 the random and
excitation-qualified protocols separate (88% vs 92%), while for N>=5 all
evaluated subsets recover the complete law. No simulated or placeholder values
are used.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import MODEL_LABELS, MULTISEED, apply_style, style_axes


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "fig5_identification_efficiency"
MATCH_FIELDS = ("seed", "subset_id", "protocol", "sample_size", "repeat")

MODELS = ["Pdiag finite", "TP-GMM SE(2)", "Full operator", "Generic RBF"]
MARKERS = {
    "Pdiag finite": "o",
    "TP-GMM SE(2)": "^",
    "Full operator": "s",
    "Generic RBF": "D",
}
LINESTYLES = {
    "Pdiag finite": "-",
    "TP-GMM SE(2)": "-.",
    "Full operator": "--",
    "Generic RBF": ":",
}
PLOT_COLORS = {
    "Pdiag finite": "#4037A3",   # ours (deep purple)
    "TP-GMM SE(2)": "#377EB8",   # blue
    "Full operator": "#8C8C8C",  # gray
    "Generic RBF": "#B0B0B0",    # lighter gray
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_success(row: dict[str, str]) -> bool:
    return row.get("fit_success", "True").strip().lower() in {"true", "1"}


def _match_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in MATCH_FIELDS)


def load_matched_fits() -> list[dict[str, str]]:
    """Load the four models on exactly the TP-GMM-matched seed/subset cells."""
    rows: list[dict[str, str]] = []
    tpgmm_root = MULTISEED / "tpgmm_fewshot"
    for tpgmm_path in sorted(tpgmm_root.glob("seed_*/tpgmm_fewshot_results.csv")):
        tpgmm_rows = _read_csv(tpgmm_path)
        selected_keys = {_match_key(row) for row in tpgmm_rows}
        if len(selected_keys) != len(tpgmm_rows):
            raise ValueError(f"Duplicate TP-GMM match key in {tpgmm_path}")
        fewshot_path = MULTISEED / "fewshot" / tpgmm_path.parent.name / "fewshot_results.csv"
        fewshot_rows = [
            row
            for row in _read_csv(fewshot_path)
            if _match_key(row) in selected_keys
        ]
        indexed: dict[tuple[tuple[str, ...], str], dict[str, str]] = {}
        for row in [*fewshot_rows, *tpgmm_rows]:
            index_key = (_match_key(row), row["model"])
            if index_key in indexed:
                raise ValueError(f"Duplicate model/key row: {index_key}")
            indexed[index_key] = row

        expected_pairs = {(key, model) for key in selected_keys for model in MODELS}
        observed_pairs = set(indexed)
        if observed_pairs != expected_pairs:
            missing = sorted(expected_pairs - observed_pairs)
            extra = sorted(observed_pairs - expected_pairs)
            raise ValueError(f"Matched key mismatch; missing={missing[:4]}, extra={extra[:4]}")
        failed = [pair for pair, row in indexed.items() if not _is_success(row)]
        if failed:
            raise ValueError(f"Matched fits contain failures: {failed[:4]}")
        rows.extend(indexed[pair] for pair in sorted(expected_pairs))

    if not rows:
        raise ValueError("No matched seed-subset fits were found")
    return rows


def summarize_matched(
    rows: list[dict[str, str]], metric: str
) -> dict[str, dict[int, tuple[float, float, int]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            grouped[(row["model"], int(row["sample_size"]))].append(value)

    summary: dict[str, dict[int, tuple[float, float, int]]] = defaultdict(dict)
    for (model, sample_size), values in grouped.items():
        array = np.asarray(values, dtype=float)
        sem = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
        summary[model][sample_size] = (float(array.mean()), sem, int(array.size))
    return summary


def complete_law_recovery() -> list[dict[str, object]]:
    """Return exact Pdiag complete-law pass rates from all frozen fits.

    The denominator is every frozen seed--subset cell, including the retained
    optimizer-limit fit.  The N=30 full-data fit has protocol ``all`` and is
    intentionally not duplicated as random and qualified evidence.
    """
    records: list[dict[str, object]] = []
    for path in sorted((MULTISEED / "fewshot").glob("seed_*/fewshot_results.csv")):
        for row in _read_csv(path):
            if row["model"] != "Pdiag finite":
                continue
            passed = (
                float(row["g_translation_final"]) > 0.90
                and float(row["g_yaw_preclear"]) > 0.90
                and abs(float(row["g_yaw_final"])) < 0.10
            )
            records.append(
                {
                    "protocol": row["protocol"],
                    "sample_size": int(row["sample_size"]),
                    "seed": int(row["seed"]),
                    "subset_id": int(row["subset_id"]),
                    "law_pass": int(passed),
                }
            )
    if len(records) != 605:
        raise ValueError(f"Expected 605 frozen Pdiag fits, found {len(records)}")

    by_cell: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_cell[(str(record["protocol"]), int(record["sample_size"]))].append(record)

    summary: list[dict[str, object]] = []
    for (protocol, sample_size), cell in sorted(by_cell.items(), key=lambda item: (item[0][1], item[0][0])):
        pass_count = sum(int(row["law_pass"]) for row in cell)
        summary.append(
            {
                "panel": "c",
                "metric": "complete_generator_law_recovery_rate",
                "protocol": protocol,
                "sample_size": sample_size,
                "pass_count": pass_count,
                "fit_count": len(cell),
                "recovery_rate": pass_count / len(cell),
            }
        )
    return summary


def plot_efficiency_panel(
    ax: plt.Axes,
    summary: dict[str, dict[int, tuple[float, float, int]]],
    ylabel: str,
    title: str,
    ylim: tuple[float, float],
    yticks: list[float],
) -> None:
    for model in MODELS:
        ns = np.asarray(sorted(summary[model]), dtype=float)
        means = np.asarray([summary[model][int(n)][0] for n in ns])
        sems = np.asarray([summary[model][int(n)][1] for n in ns])
        ax.errorbar(
            ns,
            means,
            yerr=sems,
            color=PLOT_COLORS[model],
            linestyle=LINESTYLES[model],
            marker=MARKERS[model],
            markersize=4.0,
            markeredgewidth=0,
            linewidth=1.7 if model == "Pdiag finite" else 1.15,
            elinewidth=0.8,
            capsize=1.8,
            ecolor=(0.0, 0.0, 0.0, 0.55),
            zorder=4 if model == "Pdiag finite" else 3,
        )
    ax.set_xlim(3.5, 31.5)
    ax.set_xticks([5, 10, 20, 30])
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel(r"training interventions $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="semibold", pad=6)
    ax.grid(axis="y", color="0.88", linestyle=(0, (2, 2)), linewidth=0.5)
    style_axes(ax)
    ax.tick_params(labelsize=7.0)


def write_source_data(
    generator_summary: dict[str, dict[int, tuple[float, float, int]]],
    task_summary: dict[str, dict[int, tuple[float, float, int]]],
    recovery_rows: list[dict[str, object]],
) -> None:
    fields = [
        "panel",
        "metric",
        "model",
        "sample_size",
        "mean",
        "sem",
        "cell_count",
        "protocol",
        "pass_count",
        "fit_count",
        "recovery_rate",
    ]
    with OUTPUT.with_name(OUTPUT.name + "_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for panel, metric, summary in [
            ("a", "generator_rmse", generator_summary),
            ("b", "task_error_mean_mm_equiv", task_summary),
        ]:
            for model in MODELS:
                for sample_size, (mean, sem, count) in sorted(summary[model].items()):
                    writer.writerow(
                        {
                            "panel": panel,
                            "metric": metric,
                            "model": model,
                            "sample_size": sample_size,
                            "mean": mean,
                            "sem": sem,
                            "cell_count": count,
                        }
                    )
        for row in recovery_rows:
            writer.writerow(row)


def write_matched_raw(rows: list[dict[str, str]]) -> None:
    """Export all 320 underlying matched observations for auditability."""
    fields = [
        *MATCH_FIELDS,
        "model",
        "fit_success",
        "generator_rmse",
        "task_error_mean_mm_equiv",
    ]
    path = OUTPUT.with_name(OUTPUT.name + "_matched_raw.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (_match_key(item), item["model"])):
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    apply_style()
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

    matched = load_matched_fits()
    generator_summary = summarize_matched(matched, "generator_rmse")
    task_summary = summarize_matched(matched, "task_error_mean_mm_equiv")
    recovery_rows = complete_law_recovery()
    expected_recovery = {
        ("random", 3): (44, 50),
        ("qualified", 3): (46, 50),
        ("random", 5): (50, 50),
        ("qualified", 5): (50, 50),
        ("random", 8): (50, 50),
        ("qualified", 8): (50, 50),
        ("random", 10): (50, 50),
        ("qualified", 10): (50, 50),
        ("random", 15): (50, 50),
        ("qualified", 15): (50, 50),
        ("random", 20): (50, 50),
        ("qualified", 20): (50, 50),
        ("all", 30): (5, 5),
    }
    observed_recovery = {
        (str(row["protocol"]), int(row["sample_size"])): (
            int(row["pass_count"]), int(row["fit_count"])
        )
        for row in recovery_rows
    }
    if observed_recovery != expected_recovery:
        raise ValueError(f"Unexpected complete-law recovery summary: {observed_recovery}")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    plot_efficiency_panel(
        axes[0],
        generator_summary,
        r"generator-law error $E_\alpha$",
        "(a)  Generator-law error",
        (0.15, 0.62),
        [0.2, 0.3, 0.4, 0.5, 0.6],
    )
    plot_efficiency_panel(
        axes[1],
        task_summary,
        r"$E_{\rm task}$ (mm-equiv.)",
        "(b)  Task reproduction error",
        (1.8, 8.8),
        [2, 4, 6, 8],
    )

    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.25, wspace=0.30)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=PLOT_COLORS[model],
            linestyle=LINESTYLES[model],
            marker=MARKERS[model],
            markersize=4.0,
            linewidth=1.7 if model == "Pdiag finite" else 1.15,
            markeredgewidth=0,
            label=MODEL_LABELS[model],
        )
        for model in MODELS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.035),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    write_source_data(generator_summary, task_summary, recovery_rows)
    write_matched_raw(matched)

    counts = sorted(
        {
            (sample_size, count)
            for sample_size, (_, _, count) in generator_summary["Pdiag finite"].items()
        }
    )
    print(f"saved {OUTPUT}.{{png,pdf,svg,tiff}}")
    print(f"matched seed-subset cell counts by N: {counts}")
    print("complete-law recovery (not plotted):", observed_recovery)


if __name__ == "__main__":
    main()
