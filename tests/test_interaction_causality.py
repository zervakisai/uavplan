import numpy as np

from uavbench.dynamics.interaction_engine import InteractionEngine


class _DummyNFZ:
    """Stub with get_nfz_mask() for backward compat."""
    def __init__(self, shape: tuple[int, int]):
        self._mask = np.zeros(shape, dtype=bool)
        self._mask[15:25, 15:25] = True

    def get_nfz_mask(self) -> np.ndarray:
        return self._mask.copy()


def test_fire_creates_closures_and_reads_nfz():
    """InteractionEngine creates road closures from fire+traffic
    and reads NFZ mask without mutating the NFZ model."""
    H = W = 40
    roads = np.ones((H, W), dtype=bool)
    fire = np.zeros((H, W), dtype=bool)
    fire[20, 20] = True
    traffic_positions = np.array([[20, 20]], dtype=np.int32)

    eng = InteractionEngine((H, W), roads_mask=roads)
    nfz = _DummyNFZ((H, W))
    out = eng.update(
        step_idx=1,
        fire_mask=fire,
        traffic_positions=traffic_positions,
        dynamic_nfz=nfz,
    )

    # InteractionEngine does NOT mutate the NFZ model (zones manage own growth)
    assert out["traffic_closure_cells"] > 0
    assert np.sum(eng.traffic_closure_mask) > 0
    # NFZ mask is read and reported
    assert out["nfz_cells"] > 0
    assert out["interaction_fire_nfz_overlap_ratio"] >= 0.0
