"""Smoke test for the PURE cacher: raw PNG tree -> zarr store -> loaded window.

One test per stage of the bridge, over a synthetic four-frame recording. The
point is that ``tools/cache_pure.py`` writes something ``PUREDataset`` accepts —
the loader contract itself is already covered by ``test_neckflix_zarr.py``.
"""

import json

import numpy as np
import pytest
import zarr
from PIL import Image

from dataset.data_loader.neckflix_config import normalise_participant
from dataset.data_loader.PURELoader import PUREDataset
from tools.cache_pure import write_store

FPS_NS = 33_333_333          # ~30 fps in nanoseconds
N_FRAMES = 8
HW = (4, 6)                  # (H, W), deliberately non-square


def make_raw_recording(root, name="01-01", *, nested=False):
    """A miniature PURE recording: PNG sequence plus its JSON sidecar."""
    recording_dir = root / name
    frame_dir = recording_dir / name if nested else recording_dir
    frame_dir.mkdir(parents=True)

    start = 1_392_643_993_642_815_000
    image_ns = [start + i * FPS_NS for i in range(N_FRAMES)]
    for i, stamp in enumerate(image_ns):
        # A distinct constant per frame and channel, so a transposed or
        # channel-swapped write is visible rather than plausible.
        pixels = np.zeros((*HW, 3), dtype=np.uint8)
        pixels[..., 0], pixels[..., 1], pixels[..., 2] = i, i + 50, i + 100
        Image.fromarray(pixels).save(frame_dir / f"Image{stamp}.png")

    # The oximeter runs at twice the frame rate, over the same span.
    wave_ns = [start + i * (FPS_NS // 2) for i in range(2 * N_FRAMES)]
    sidecar = {
        "/FullPackage": [
            {"Timestamp": ts, "Value": {"waveform": float(i)}, "FrameID": ""}
            for i, ts in enumerate(wave_ns)
        ],
        "/Image": [
            {"Timestamp": ts, "Value": {}, "FrameID": "/CameraFrame"}
            for ts in image_ns
        ],
    }
    (root / f"{name}.json").write_text(json.dumps(sidecar))
    return recording_dir


def cache_cfg(cache_dir, **overrides):
    cfg = {
        "cache_dir": str(cache_dir),
        "channels": ["R", "G", "B"],
        "labels": ["PPG"],
        "target_fps": 30.0,
        "window_seconds": 4 / 30,
        "stride_seconds": 4 / 30,
        "window_size": 4,
        "window_stride": 4,
        "random_windows": False,
        "filters": {},
        "label_norms": {"PPG": "zscore"},
        "allow_missing": False,
        "min_channels": 1,
        "min_labels": 1,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def cached(tmp_path):
    """``(cache_dir, store_path)`` for one cached synthetic recording."""
    raw = tmp_path / "raw"
    raw.mkdir()
    make_raw_recording(raw)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    store = write_store(raw / "01-01", cache_dir, align="timestamp", overwrite=False)
    return cache_dir, store


def test_store_satisfies_the_admission_gate_and_layout(cached):
    _, store = cached
    root = zarr.open_group(str(store), mode="r")
    assert root.attrs["complete"] is True          # identity, not a string
    assert root.attrs["tool_version"] == "1.0.0"
    assert root.attrs["recording"] == "01-01"
    assert root.attrs["setup_name"] == "steady"

    video = root["1"]["rgb"]["video"]
    assert video["frames"].shape == (3, N_FRAMES, *HW)
    assert video["frames"].dtype == np.uint8
    assert video.attrs["num_frames"] == N_FRAMES   # required by the loader
    assert video.attrs["fps"] == pytest.approx(30.0, abs=0.01)
    # The trace group is 'ppg': config TRACES are canonicalised (BVP -> PPG)
    # before the loader looks it up, so a 'bvp' group would never be found.
    assert root["1"]["rgb"]["ppg"]["data"].shape == (N_FRAMES,)


def test_participant_attr_matches_the_normalised_loso_id(cached):
    _, store = cached
    stored = zarr.open_group(str(store), mode="r").attrs["participant"]
    # Every spelling a config or the CLI may use must land on the stored attr,
    # or a LOSO fold silently selects nothing.
    assert stored == "001"
    assert {normalise_participant(p) for p in ("P01", "01", 1)} == {stored}


def test_frames_keep_raw_rgb_order(cached):
    _, store = cached
    frames = np.asarray(
        zarr.open_group(str(store), mode="r")["1"]["rgb"]["video"]["frames"])
    for i in range(N_FRAMES):
        assert (frames[0, i] == i).all()          # R
        assert (frames[1, i] == i + 50).all()     # G
        assert (frames[2, i] == i + 100).all()    # B


def test_trace_is_resampled_to_one_sample_per_frame(cached):
    _, store = cached
    trace = np.asarray(zarr.open_group(str(store), mode="r")["1"]["rgb"]["ppg"]["data"])
    # The synthetic waveform is the ramp 0..2N-1 at twice the frame rate over
    # the same span, so frame i falls on waveform sample 2i.
    assert trace == pytest.approx(np.arange(N_FRAMES) * 2.0)


def test_nested_and_flat_frame_layouts_agree(tmp_path):
    traces, frames = [], []
    for nested in (False, True):
        raw = tmp_path / f"raw_{nested}"
        raw.mkdir()
        make_raw_recording(raw, nested=nested)
        cache_dir = tmp_path / f"cache_{nested}"
        cache_dir.mkdir()
        store = write_store(raw / "01-01", cache_dir, align="timestamp", overwrite=False)
        root = zarr.open_group(str(store), mode="r")["1"]["rgb"]
        traces.append(np.asarray(root["ppg"]["data"]))
        frames.append(np.asarray(root["video"]["frames"]))
    assert np.array_equal(frames[0], frames[1])
    assert np.array_equal(traces[0], traces[1])


def test_cached_store_loads_through_pure_dataset(cached):
    cache_dir, _ = cached
    dataset = PUREDataset(cache_cfg(cache_dir))
    assert dataset.samples == [("01-01", "1")]
    assert len(dataset) == N_FRAMES // 4

    item = dataset[0]
    assert tuple(item["frames"]["R"].shape) == (1, 4, *HW)
    assert tuple(item["labels"]["PPG"].shape) == (4,)
    assert bool(item["label_mask"]["PPG"])
    assert all(bool(m) for m in item["channel_mask"].values())
    assert item["metadata"]["recording_id"] == "01-01"


def test_channels_pure_lacks_arrive_zero_filled(cached):
    """A Neckflix-shaped interface (R,G,B,I,D) still runs on RGB-only PURE."""
    cache_dir, _ = cached
    with pytest.warns(UserWarning, match="not provided by PUREDataset"):
        dataset = PUREDataset(cache_cfg(cache_dir, channels=["R", "G", "B", "I", "D"]))
    item = dataset[0]
    for channel in ("I", "D"):
        assert not bool(item["channel_mask"][channel])
        assert (item["frames"][channel] == 0).all()
    assert bool(item["channel_mask"]["R"])


def test_interrupted_write_leaves_an_inadmissible_store(tmp_path, monkeypatch):
    """A crash mid-write must not leave a store the loader would accept.

    ``complete: true`` goes on last, so a half-written store fails the
    admission gate instead of training on a silently truncated recording.
    """
    import tools.cache_pure as cache_pure

    raw = tmp_path / "raw"
    raw.mkdir()
    make_raw_recording(raw)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    # Fails inside the frame-writing loop: the store group exists by then, but
    # no root attrs have been written.
    monkeypatch.setattr(cache_pure, "tqdm", explode)
    with pytest.raises(RuntimeError, match="boom"):
        cache_pure.write_store(raw / "01-01", cache_dir, align="timestamp",
                               overwrite=False)

    store = cache_dir / "01-01.zarr"
    assert store.exists(), "expected a partially written store to exist"
    assert zarr.open_group(str(store), mode="r").attrs.get("complete") is not True

    with pytest.warns(UserWarning, match="no 'complete: true' root attr"):
        with pytest.raises(RuntimeError, match="No usable zarr stores"):
            PUREDataset(cache_cfg(cache_dir))
