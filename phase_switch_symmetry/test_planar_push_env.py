import numpy as np

import phase_switch_symmetry_planar_push_env as pp


def _bare_env(causal_delta=None):
    """Construct a PlanarPushEnv without the heavy BaseEnv scene init, enough to
    exercise the pure frame-math helpers (_to_world / block_goal_position)."""
    env = object.__new__(pp.PlanarPushEnv)
    env.task_anchor = pp.TARGET_POS
    env.Q_mat = np.eye(3)
    env.causal_delta = (
        np.zeros(6, dtype=np.float64) if causal_delta is None else np.asarray(causal_delta, dtype=np.float64)
    )
    return env


def test_block_footprint_is_non_square():
    # Spec §4.1: rectangular, clearly non-square footprint, 2:1 aspect.
    assert pp.BLOCK_HALF_X == 0.02
    assert pp.BLOCK_HALF_Y == 0.01
    assert pp.BLOCK_HALF_X != pp.BLOCK_HALF_Y


def test_to_world_identity_maps_local_to_anchor_plus_offset():
    env = _bare_env()
    out = env._to_world([0.01, -0.02, 0.03])
    assert np.allclose(out, pp.TARGET_POS + np.array([0.01, -0.02, 0.03]))


def test_block_goal_position_is_in_plane_only():
    env = _bare_env(causal_delta=[0.005, -0.004, 0.012, 0.1, 0.2, 0.3])
    p = env.block_goal_position()
    # in-plane follows du,dv; z pinned to block height (dw ignored)
    assert np.allclose(p[:2], pp.TARGET_POS[:2] + np.array([0.005, -0.004]))
    assert np.isclose(p[2], pp.BLOCK_HALF_Z)


def test_heading_of_recovers_yaw_exactly():
    from transforms3d.quaternions import quat2mat
    for yaw in [-0.5, 0.0, 0.3]:
        q = pp._local_quat(0.0, 0.0, yaw)
        assert np.isclose(pp._heading_of(q), yaw, atol=1e-9)


def test_wrap_pi():
    assert np.isclose(pp._wrap_pi(0.5), 0.5)
    assert np.isclose(pp._wrap_pi(np.pi + 0.2), -np.pi + 0.2)
    assert np.isclose(pp._wrap_pi(-np.pi - 0.2), np.pi - 0.2)


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in _tests:
        fn()
        print("PASS", fn.__name__)
    print(f"ALL {len(_tests)} UNIT TESTS PASS")
