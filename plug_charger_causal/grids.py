from __future__ import annotations

import math

import numpy as np


def row(generator: str, dx: float, dy: float, dyaw: float) -> dict:
    return {
        "generator": generator,
        "causal_delta": [dx, dy, dyaw],
        "delta_x_m": dx,
        "delta_y_m": dy,
        "delta_yaw_rad": dyaw,
        "delta_x_mm": dx * 1000.0,
        "delta_y_mm": dy * 1000.0,
        "delta_yaw_deg": math.degrees(dyaw),
    }


def isolated_grid() -> list[dict]:
    x_mm = [-20, -10, 0, 10, 20]
    y_mm = [-20, -10, 0, 10, 20]
    yaw_deg = [-15, -7.5, 0, 7.5, 15]

    rows = []
    for value in x_mm:
        rows.append(row("dx", value / 1000.0, 0.0, 0.0))
    for value in y_mm:
        if value != 0:
            rows.append(row("dy", 0.0, value / 1000.0, 0.0))
    for value in yaw_deg:
        if value != 0:
            rows.append(row("dyaw", 0.0, 0.0, math.radians(value)))
    return rows


def smoke_grid() -> list[dict]:
    return [
        row("dx", 0.02, 0.0, 0.0),
        row("dy", 0.0, 0.02, 0.0),
        row("dyaw", 0.0, 0.0, math.radians(15)),
    ]


def mixed_training_grid(num_samples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_samples):
        dx = float(rng.uniform(-0.01, 0.01))
        dy = float(rng.uniform(-0.01, 0.01))
        dyaw = float(rng.uniform(math.radians(-7.5), math.radians(7.5)))
        rows.append(row("mixed", dx, dy, dyaw))
    return rows
