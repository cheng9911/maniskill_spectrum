from __future__ import annotations

"""Shared task specs for the controlled LIBERO relation suite."""

from dataclasses import dataclass


GENERATOR_BASIS = ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"]
PHASE_CODES = (3, 4, 5, 6)
PHASE_NAMES = ("reach", "move", "hold", "retract")
SEEDS = [20260818, 20270818, 20280818]


@dataclass(frozen=True)
class LiberoRelationSpec:
    task_key: str
    suite_name: str
    task_id: int
    language: str
    family: str
    oracle_selector: tuple[int, int, int, int, int, int]
    nominal_pose6: tuple[float, float, float, float, float, float]
    body_name: str
    joint_name: str | None = None
    free_joint_name: str | None = None
    control: str = "free_pose"
    joint_zero: float = 0.0
    joint_sign: float = 1.0
    notes: str = ""


LIBERO_RELATION_SPECS = (
    LiberoRelationSpec(
        task_key="drawer_middle_open",
        suite_name="libero_goal",
        task_id=0,
        language="open the middle drawer of the cabinet",
        family="prismatic_sliding",
        oracle_selector=(1, 0, 0, 0, 0, 0),
        nominal_pose6=(0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
        body_name="wooden_cabinet_1_cabinet_middle",
        joint_name="wooden_cabinet_1_middle_level",
        control="prismatic",
        joint_zero=0.0,
        joint_sign=-1.0,
        notes="drawer rail translation only",
    ),
    LiberoRelationSpec(
        task_key="stove_knob_turn",
        suite_name="libero_goal",
        task_id=7,
        language="turn on the stove",
        family="revolute_knob",
        oracle_selector=(0, 0, 0, 0, 0, 1),
        nominal_pose6=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        body_name="flat_stove_1_button",
        joint_name="flat_stove_1_button",
        control="revolute_yaw",
        joint_zero=0.0,
        joint_sign=1.0,
        notes="button hinge rotation only",
    ),
    LiberoRelationSpec(
        task_key="microwave_door_revolute",
        suite_name="libero_10",
        task_id=9,
        language="put the yellow and white mug in the microwave and close it",
        family="revolute_door",
        oracle_selector=(0, 0, 0, 0, 0, 1),
        nominal_pose6=(0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        body_name="microwave_1_microdoorroot",
        joint_name="microwave_1_microjoint",
        control="revolute_yaw",
        joint_zero=0.0,
        joint_sign=1.0,
        notes="door hinge subtask scored as a revolute relation",
    ),
    LiberoRelationSpec(
        task_key="plate_front_push",
        suite_name="libero_goal",
        task_id=5,
        language="push the plate to the front of the stove",
        family="planar_one_axis_push",
        oracle_selector=(1, 0, 0, 0, 0, 0),
        nominal_pose6=(0.10, 0.0, 0.0, 0.0, 0.0, 0.0),
        body_name="plate_1_main",
        free_joint_name="plate_1_joint0",
        notes="front translation tracked; lateral/vertical/rotations suppressed",
    ),
    LiberoRelationSpec(
        task_key="bowl_on_stove",
        suite_name="libero_goal",
        task_id=1,
        language="put the bowl on the stove",
        family="support_placement",
        oracle_selector=(1, 1, 1, 0, 0, 0),
        nominal_pose6=(0.08, 0.02, 0.06, 0.0, 0.0, 0.0),
        body_name="akita_black_bowl_1_main",
        free_joint_name="akita_black_bowl_1_joint0",
        notes="target translation tracked; object kept upright",
    ),
    LiberoRelationSpec(
        task_key="bowl_on_plate",
        suite_name="libero_goal",
        task_id=8,
        language="put the bowl on the plate",
        family="stacking_support",
        oracle_selector=(1, 1, 1, 0, 0, 0),
        nominal_pose6=(0.07, -0.02, 0.055, 0.0, 0.0, 0.0),
        body_name="akita_black_bowl_1_main",
        free_joint_name="akita_black_bowl_1_joint0",
        notes="stacking target translation tracked; yaw free",
    ),
    LiberoRelationSpec(
        task_key="cream_cheese_in_bowl",
        suite_name="libero_goal",
        task_id=6,
        language="put the cream cheese in the bowl",
        family="container_in",
        oracle_selector=(1, 1, 1, 0, 0, 0),
        nominal_pose6=(0.07, 0.02, 0.045, 0.0, 0.0, 0.0),
        body_name="cream_cheese_1_main",
        free_joint_name="cream_cheese_1_joint0",
        notes="container-relative translation tracked; rotations suppressed",
    ),
    LiberoRelationSpec(
        task_key="wine_bottle_on_rack",
        suite_name="libero_goal",
        task_id=9,
        language="put the wine bottle on the rack",
        family="rack_slot_placement",
        oracle_selector=(1, 1, 1, 0, 0, 1),
        nominal_pose6=(0.08, 0.0, 0.08, 0.0, 0.0, 0.0),
        body_name="wine_bottle_1_main",
        free_joint_name="wine_bottle_1_joint0",
        notes="slot translation and axial heading tracked; out-of-plane tilt suppressed",
    ),
    LiberoRelationSpec(
        task_key="wine_bottle_on_cabinet",
        suite_name="libero_goal",
        task_id=2,
        language="put the wine bottle on top of the cabinet",
        family="upright_support_placement",
        oracle_selector=(1, 1, 1, 0, 0, 0),
        nominal_pose6=(0.08, 0.0, 0.10, 0.0, 0.0, 0.0),
        body_name="wine_bottle_1_main",
        free_joint_name="wine_bottle_1_joint0",
        notes="upright placement translation tracked; yaw ignored",
    ),
    LiberoRelationSpec(
        task_key="moka_pot_on_stove",
        suite_name="libero_10",
        task_id=2,
        language="turn on the stove and put the moka pot on it",
        family="compound_support_subtask",
        oracle_selector=(1, 1, 1, 0, 0, 0),
        nominal_pose6=(0.09, -0.02, 0.07, 0.0, 0.0, 0.0),
        body_name="moka_pot_1_main",
        free_joint_name="moka_pot_1_joint0",
        notes="placement subtask from a compound LIBERO task",
    ),
)


def specs_by_key() -> dict[str, LiberoRelationSpec]:
    return {spec.task_key: spec for spec in LIBERO_RELATION_SPECS}
