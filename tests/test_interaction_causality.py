import numpy as np

from uavbench.dynamics.interaction_engine import InteractionEngine


class _DummyNFZ:
    def __init__(self):
        self.expansion_rate = 1.0


def test_fire_increases_nfz_and_creates_closures():
    H = W = 40
    roads = np.ones((H, W), dtype=bool)
    fire = np.zeros((H, W), dtype=bool)
    fire[20, 20] = True
    traffic_positions = np.array([[20, 20]], dtype=np.int32)

    eng = InteractionEngine((H, W), roads_mask=roads)
    nfz = _DummyNFZ()
    out = eng.update(
        step_idx=1,
        fire_mask=fire,
        traffic_positions=traffic_positions,
        dynamic_nfz=nfz,
    )

    assert nfz.expansion_rate > 1.0
    assert out["traffic_closure_cells"] > 0
    assert np.sum(eng.traffic_closure_mask) > 0
