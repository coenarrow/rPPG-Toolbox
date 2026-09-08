import torch

from src.interface import load_interface
from src.models import build_model, load_model_config

INTERFACE = "configs/interfaces/physnet_interface.yaml"


def test_physnet_forward_matches_the_contract():
    interface = load_interface(INTERFACE)
    model = build_model(load_model_config("physnet", interface), interface)
    B, T = 2, interface.window_frames
    H, W = interface.RESIZE.H, interface.RESIZE.W
    batch = {"frames": {ch: {prep: torch.zeros(B, T, H, W)
                             for prep in interface.INPUT_PREPROCESSING}
                        for ch in interface.CHANNELS}}
    out = model(batch)
    assert set(out["predictions"]) == set(interface.TRACES)
    assert all(p.shape == (B, T) for p in out["predictions"].values())
    assert len(model.output_layers()) == len(interface.TRACES)
