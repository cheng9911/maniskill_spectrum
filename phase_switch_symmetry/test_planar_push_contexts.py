import json
from pathlib import Path

import numpy as np


def test_manifest_shape():
    path = Path(
        "phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json"
    )
    m = json.loads(path.read_text())
    conds = m["conditions"]
    assert len(conds) == 75
    assert sum(c["generator"] == "baseline" for c in conds) == 1
    assert sum(c["generator"] == "mixed" for c in conds) == 60
    assert (
        sum(c["generator"] in {"du", "dv", "dw", "roll", "pitch", "yaw"} for c in conds)
        == 14
    )
    ids = [c["condition_id"] for c in conds]
    assert ids == list(range(75))
    assert all(np.asarray(c["causal_delta"]).shape == (6,) for c in conds)
    assert all(np.isfinite(c["causal_delta"]).all() for c in conds)


if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in _tests:
        fn()
        print("PASS", fn.__name__)
    print(f"ALL {len(_tests)} UNIT TESTS PASS")
