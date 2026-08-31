"""The upstream tuple contract must still work after the dict-contract retrofit.

PhysMamba became a ``DictModel`` and the shared post-processing was rewritten,
both of which the non-Neckflix datasets (PURE, UBFC-rPPG, ...) sit on top of.
Those datasets are not available here, so this drives the real
``PhysMambaTrainer`` over a synthetic loader that emits exactly the upstream
``(frames, label, filename, chunk_id)`` tuple.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from neural_methods.trainer.PhysMambaTrainer import PhysMambaTrainer

FS = 30
FRAMES = 32
SIZE = 32


def _ns(**kwargs):
    """A nested attribute namespace — the shape the legacy trainers read.

    The yacs tree died with the config redesign; the legacy tuple-contract
    trainers are dead code on their Phase 6 schedule, so this test hands them
    exactly the keys they read rather than resurrecting a schema for them.
    """
    return SimpleNamespace(**kwargs)


class TupleContractDataset(Dataset):
    """``(frames (C,T,H,W), label (T,), filename, chunk_id)`` — the upstream shape."""

    def __init__(self, n_clips=4, subjects=("subject1", "subject2")):
        self.items = []
        rng = np.random.default_rng(0)
        t = np.arange(FRAMES) / FS
        for i in range(n_clips):
            pulse = np.sin(2 * np.pi * 1.2 * t + i)
            frames = (pulse[None, :, None, None]
                      + rng.normal(0, 0.1, (3, FRAMES, SIZE, SIZE))).astype(np.float32)
            self.items.append((torch.from_numpy(frames),
                               torch.from_numpy(pulse.astype(np.float32)),
                               subjects[i % len(subjects)], i))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@pytest.fixture
def legacy_config(tmp_path):
    return _ns(
        TOOLBOX_MODE="train_and_test",
        DEVICE="cpu",
        LOG=_ns(PATH=str(tmp_path)),
        MODEL=_ns(NAME="PhysMamba", MODEL_DIR=str(tmp_path / "models")),
        TRAIN=_ns(
            EPOCHS=1,
            BATCH_SIZE=2,
            LR=1e-3,
            MODEL_FILE_NAME="legacy_physmamba",
            PLOT_LOSSES_AND_LR=False,
            DATA=_ns(FS=FS, PREPROCESS=_ns(LABEL_TYPE="Standardized")),
        ),
        TEST=_ns(
            USE_LAST_EPOCH=True,
            METRICS=["MAE", "RMSE", "MACC"],
            OUTPUT_SAVE_DIR=str(tmp_path / "outputs"),
            DATA=_ns(FS=FS, EXP_DATA_NAME="legacy", DATASET="Synthetic",
                     PREPROCESS=_ns(LABEL_TYPE="Standardized")),
        ),
        INFERENCE=_ns(
            EVALUATION_METHOD="FFT",
            EVALUATION_WINDOW=_ns(USE_SMALLER_WINDOW=False, WINDOW_SIZE=10),
        ),
    )


def test_physmamba_still_returns_the_legacy_tensor_shape():
    """(B, C, T, H, W) in, (B, T) out — what the tuple-contract trainers expect."""
    from neural_methods.model.PhysMamba import PhysMamba
    with torch.no_grad():
        out = PhysMamba()(torch.randn(2, 3, FRAMES, SIZE, SIZE))
    assert out.shape == (2, FRAMES)
    assert torch.isfinite(out).all()


def test_legacy_trainer_trains_and_tests(legacy_config, tmp_path, capsys):
    loaders = {
        "train": DataLoader(TupleContractDataset(), batch_size=2, shuffle=False),
        "valid": None,
        "test": DataLoader(TupleContractDataset(), batch_size=2, shuffle=False),
    }
    trainer = PhysMambaTrainer(legacy_config, loaders, rank=0, world_size=1, debug=False)
    trainer.train(loaders)
    assert (tmp_path / "models" / "legacy_physmamba_Epoch0.pth").exists()

    trainer.test(loaders)
    printed = capsys.readouterr().out
    assert "FFT MAE" in printed
    assert (tmp_path / "outputs" / "legacy_physmamba_outputs.pickle").exists()


def test_legacy_metrics_path_survives_the_post_processing_rewrite(legacy_config):
    """evaluation.metrics feeds _detrend and _compute_macc, both rewritten."""
    from evaluation.metrics import calculate_metrics

    t = np.arange(FRAMES * 8) / FS
    wave = torch.from_numpy(np.sin(2 * np.pi * 1.2 * t).astype(np.float32))
    chunks = {i: wave[i * FRAMES:(i + 1) * FRAMES] for i in range(8)}
    calculate_metrics({"subject1": chunks}, {"subject1": chunks}, legacy_config)
