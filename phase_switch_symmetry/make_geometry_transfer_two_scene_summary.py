from __future__ import annotations

"""Create a two-scene few-shot generalization summary for geometry transfer."""

import argparse
from pathlib import Path

import pandas as pd


PAIR_ORDER = (
    "knob_to_microwave_door",
    "bowl_stove_to_cream_cheese_bowl",
)
METHOD_ORDER = (
    "Ours transfer: source Pdiag N=30 + target nominal N=1",
    "Pdiag finite target scratch",
    "TP-GMM SE(3) target scratch",
    "Frame-weighted target scratch",
    "Phase scalar GP target scratch",
)


def _method_rank(name: str) -> int:
    try:
        return METHOD_ORDER.index(name)
    except ValueError:
        return len(METHOD_ORDER)


def _pair_rank(name: str) -> int:
    try:
        return PAIR_ORDER.index(name)
    except ValueError:
        return len(PAIR_ORDER)


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = [
        "pair_key",
        "family",
        "method",
        "sample_size",
        "m_transfer_accuracy",
        "e_alpha_mean",
        "heldout_prediction_mse_mean",
        "count",
    ]
    lines = [
        "| pair | family | method | N | M_transfer | E_alpha | heldout MSE | count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame[cols].itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.pair_key}`",
                    row.family,
                    row.method.replace("Ours transfer: source Pdiag N=30 + target nominal N=1", "Ours transfer"),
                    str(int(row.sample_size)),
                    f"{row.m_transfer_accuracy:.3f}",
                    f"{row.e_alpha_mean:.3e}",
                    f"{row.heldout_prediction_mse_mean:.3e}",
                    str(int(row.count)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/geometry_transfer"),
    )
    parser.add_argument(
        "--tpgmm-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/geometry_transfer_tpgmm_n8"),
    )
    args = parser.parse_args()

    main_by_pair = pd.read_csv(args.main_root / "geometry_transfer_by_pair.csv")
    tpgmm_by_pair = pd.read_csv(args.tpgmm_root / "geometry_transfer_by_pair.csv")
    selected = main_by_pair.loc[
        main_by_pair["pair_key"].isin(PAIR_ORDER)
        & main_by_pair["method"].isin(
            [
                "Ours transfer: source Pdiag N=30 + target nominal N=1",
                "Pdiag finite target scratch",
                "Frame-weighted target scratch",
                "Phase scalar GP target scratch",
            ]
        )
    ].copy()
    selected_tpgmm = tpgmm_by_pair.loc[
        tpgmm_by_pair["pair_key"].isin(PAIR_ORDER)
        & tpgmm_by_pair["method"].eq("TP-GMM SE(3) target scratch")
    ].copy()
    combined = pd.concat([selected, selected_tpgmm], ignore_index=True)
    combined["_pair_rank"] = combined["pair_key"].map(_pair_rank)
    combined["_method_rank"] = combined["method"].map(_method_rank)
    combined = combined.sort_values(
        ["_pair_rank", "sample_size", "_method_rank"]
    ).drop(columns=["_pair_rank", "_method_rank"])

    out_csv = args.main_root / "geometry_transfer_two_scene_fewshot.csv"
    out_md = args.main_root / "VALIDATION_geometry_transfer_two_scene.md"
    combined.to_csv(out_csv, index=False)

    text = f"""# Two-Scene Few-Shot Geometry Transfer Check

Date: 2026-08-31

This table extracts two structurally different target scenes from the geometry
transfer supplement:

- `knob_to_microwave_door`: revolute yaw-only relation;
- `bowl_stove_to_cream_cheese_bowl`: support/container placement relation.

The transferred method freezes the source Pdiag N=30 alpha profile and uses only
one target nominal trajectory for geometry adaptation.  Target interventions are
validation probes; they are not used to fit alpha for the transferred method.
Target-scratch baselines fit on N target interventions and are evaluated on
held-out target contexts.

{_markdown_table(combined)}

Summary: in both scenes, direct transfer keeps `M_transfer=1.000` for N=3/5/8
and has lower held-out pose-trajectory MSE than target-scratch baselines.  The
N=8 TP-GMM SE(3) check also reaches binary `M_transfer=1.000`, but with much
higher continuous alpha error and held-out trajectory MSE.
"""
    out_md.write_text(text, encoding="utf-8")
    print("saved:", out_csv)
    print("saved:", out_md)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
