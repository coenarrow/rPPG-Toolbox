"""The train -> infer -> eval chain, end to end over a synthetic zarr cache on the CPU.

One epoch of PhysNet on three participants holding one out, then the
checkpoint over that participant and over another, then a run with nobody
held out. The scripts have to leave exactly the files the README promises
where it promises them, and ``infer`` has to rebuild the run from ``model.pt``
alone, and ``eval`` has to score the records from the directory alone. The
stride is half a window, so the trace tables carry overlaps.
"""
import json
import textwrap

import numpy as np
import pandas as pd
import pytest

import src.datasets
from scripts.infer import main as infer
from scripts.eval import main as evaluate
from scripts.train import main as train
from src.outputs import META_NAME, RECORDS_DIR, WINDOWS_NAME
from src.trainer import CHECKPOINT_NAME, CONFIG_NAME, LOSS_LOG_NAME
from tests.zarr_fixtures import make_store

WINDOW = 16       # frames per window at the fixture rate
STRIDE = WINDOW // 2
FRAME_SIZE = 32
PARTICIPANTS = ("001", "002", "003")


def _write(path, text):
    path.write_text(textwrap.dedent(text))
    return path


@pytest.fixture
def train_args(tmp_path, monkeypatch):
    """The four config arguments of a CPU PhysNet run over a 3-participant cache."""
    cache = tmp_path / "cache"
    cache.mkdir()
    for pid in PARTICIPANTS:
        make_store(cache, name=f"P{pid}_S01_R1_0_D", modalities=("rgb",),
                   traces=("abp", "cvp"), num_frames=2 * WINDOW,
                   hw=(FRAME_SIZE, FRAME_SIZE))
    configs = tmp_path / "datasets"
    configs.mkdir()
    _write(configs / "smoke.yaml", f"""
        CACHED_PATH: "{cache.as_posix()}"
        FILTERS: {{}}
        """)
    monkeypatch.setattr(src.datasets, "DATASET_CONFIG_DIR", configs)
    interface = _write(tmp_path / "interface.yaml", f"""
        FS: 30.0
        UPSAMPLING: refuse
        WINDOW_SECONDS: {WINDOW / 30}
        WINDOW_STRIDE: {STRIDE / 30}
        CHANNELS: [R, G, B]
        TRACES: [ABP, CVP]
        RESIZE: {{H: {FRAME_SIZE}, W: {FRAME_SIZE}}}
        INPUT_PREPROCESSING: [DiffNormalized]
        LABEL_PREPROCESSING: {{ABP: zscore, CVP: zscore}}
        LOSS:
          ABP: {{NEGPEARSON: 1.0}}
          CVP: {{NEGPEARSON: 1.0}}
        """)
    training = _write(tmp_path / "training.yaml", """
        EPOCHS: 1
        BATCH_SIZE: 2
        OPTIMIZER: Adam
        LR: 1e-3
        WEIGHT_DECAY: 0.0
        SCHEDULER: Constant
        PRECISION: float32
        DEVICE: cpu
        NUM_WORKERS: 0
        """)
    return ["--datasets", "smoke", "--model", "physnet",
            "--interface", str(interface), "--training", str(training),
            "--runs-dir", str(tmp_path / "runs")]



def _stem(run_dir):
    """The run name without its ``_<YYYYMMDDHHMM>`` timestamp."""
    name, _, stamp = run_dir.name.rpartition("_")
    assert len(stamp) == 12 and stamp.isdigit(), run_dir.name
    return name

def _participants(records) -> set:
    return {r["metadata"]["participant"] for r in records}


def test_train_writes_the_run_and_infer_rebuilds_it(tmp_path, train_args):
    run_dir = train([*train_args, "--test-participant-dataset", "smoke",
                     "--test-participant-id", "003"])
    assert run_dir.parent == tmp_path / "runs" and _stem(run_dir) == "PHYSNET_SMOKE.003"
    assert {CHECKPOINT_NAME, CONFIG_NAME, LOSS_LOG_NAME} <= {p.name for p in run_dir.iterdir()}
    assert not (run_dir / RECORDS_DIR).exists()       # training infers nothing

    # The run directory alone: the held-out participant, records beside the checkpoint.
    records = infer([str(run_dir)])
    assert _participants(records) == {"003"} and len(records) == 3
    out = run_dir / RECORDS_DIR
    meta = json.loads((out / META_NAME).read_text(encoding="utf-8"))
    assert meta["participant"] == "003" and meta["fs"] == 30.0 and meta["n_windows"] == 3
    windows = pd.read_csv(out / WINDOWS_NAME, dtype={"participant": str})
    assert list(windows["start_frame"]) == [0, STRIDE, 2 * STRIDE]
    assert set(windows["participant"]) == {"003"} and windows["label_ABP"].all()

    # One wide table per trace: frames 0..31, three staggered window columns,
    # the label everywhere, mean/std/n over whichever windows cover a frame.
    abp = pd.read_csv(out / "P003_S01_R1_0_D" / "1" / "ABP.csv")
    assert list(abp.columns) == ["frame", "t", "label", "mean", "std", "n",
                                 "w0", f"w{STRIDE}", f"w{2 * STRIDE}"]
    assert len(abp) == 2 * WINDOW and abp["t"].iloc[-1] == pytest.approx((2 * WINDOW - 1) / 30, abs=1e-5)
    assert list(abp["n"]) == [1] * STRIDE + [2] * 2 * STRIDE + [1] * STRIDE
    assert abp["label"].notna().all()
    single = abp["n"] == 1
    assert np.allclose(abp.loc[single, "mean"],
                       abp.loc[single, ["w0", f"w{2 * STRIDE}"]].sum(axis=1))
    assert abp.loc[single, "std"].isna().all() and abp.loc[~single, "std"].notna().all()
    assert (out / "P003_S01_R1_0_D" / "1" / "CVP.csv").is_file()

    # Another participant, elsewhere: the same layout, no second config.yaml.
    out = tmp_path / "elsewhere"
    records = infer([str(run_dir), "--test-participant-dataset", "smoke",
                     "--test-participant-id", "001", "--out", str(out)])
    assert _participants(records) == {"001"}
    assert (out / WINDOWS_NAME).is_file() and (out / "P001_S01_R1_0_D" / "1" / "ABP.csv").is_file()
    assert not (out / CONFIG_NAME).exists() and not (out / RECORDS_DIR).exists()

    # The records alone: every recording scored beside its trace tables. 32
    # covered frames at 30 fps cut into 0.5 s readings gives two; the
    # two-frame remainder is dropped.
    folder = run_dir / RECORDS_DIR / "P003_S01_R1_0_D" / "1"
    assert evaluate([str(run_dir), "--reading-seconds", "0.5"]) == [folder]
    for name in ("readings.csv", "beats.csv", "rates.csv"):
        assert (folder / name).is_file(), name
    readings = pd.read_csv(folder / "readings.csv")
    assert set(readings["signal"]) == {"ABP", "CVP"}
    assert sorted(readings["reading"].unique()) == [0, 1]
    assert readings["waveform_mad"].notna().all()


def test_train_on_everyone_then_infer_must_name_a_participant(tmp_path, train_args):
    run_dir = train(train_args)
    assert run_dir.parent == tmp_path / "runs" and _stem(run_dir) == "PHYSNET_SMOKE.all"
    with pytest.raises(SystemExit):                    # nobody was held out
        infer([str(run_dir)])
    records = infer([str(run_dir), "--test-participant-dataset", "smoke",
                     "--test-participant-id", "002"])
    assert _participants(records) == {"002"}
