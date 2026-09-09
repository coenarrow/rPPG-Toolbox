"""The memory report, end to end over a synthetic zarr cache on the CPU.

One window through one real training step: the report has to name the
parameter count, the bytes the weights, gradients and optimiser state take,
the bytes one batch takes, and — off the GPU — say plainly that no device
measurement was made.
"""
import textwrap

import src.datasets
from tests.zarr_fixtures import make_store
from tools.memory_report import main

WINDOW = 16       # frames per window at the fixture rate
FRAME_SIZE = 32


def _write(path, text):
    path.write_text(textwrap.dedent(text))
    return path


def test_report_counts_the_footprint_of_one_training_step(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("P001_S01_R1_0_D", "P002_S01_R1_0_D"):
        make_store(cache, name=name, modalities=("rgb",), traces=("abp", "cvp"),
                   num_frames=2 * WINDOW, hw=(FRAME_SIZE, FRAME_SIZE))
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
        WINDOW_STRIDE: {WINDOW / 30}
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

    report = main(["--datasets", "smoke", "--model", "physnet",
                   "--interface", str(interface), "--training", str(training)])

    n = report["parameters"]
    assert n > 0
    assert report["bytes"]["weights"] == 4 * n        # float32 weights
    assert report["bytes"]["gradients"] == 4 * n      # one grad per weight
    # Adam: two float32 moments per weight plus one scalar step per tensor.
    counters = 4 * 200                            # room for 200 parameter tensors
    assert 2 * 4 * n <= report["bytes"]["optimiser_state"] <= 2 * 4 * n + counters
    # One batch: BATCH_SIZE windows of 3 channels x 1 block x T x H x W floats
    # plus the labels, stats and masks that ride with them.
    frames = 2 * 3 * WINDOW * FRAME_SIZE * FRAME_SIZE * 4
    assert report["bytes"]["batch"] >= frames
    assert report["device"]["type"] == "cpu"
    assert report["device"]["measured"] is None       # nothing to read off a CPU
    out = capsys.readouterr().out
    assert f"{n:,} parameters" in out
    assert "no device memory measurement" in out
