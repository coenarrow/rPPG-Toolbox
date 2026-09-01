"""Write the PURE dataset into the zarr cache this repo reads.

PURE ships as timestamped PNG sequences plus a JSON sidecar of pulse-oximeter
readings; the pipeline reads ``{recording}.zarr`` stores. This is the offline
bridge between the two -- the PURE equivalent of the Neckflix preprocessor
vendored at ``external/neckflix`` (``uv run --project external/neckflix
neckflix-preprocess``), kept in ``tools/`` because nothing on the training path
may write the cache. The mapping it implements is ``dataset/data_loader/PURE.md``.

    uv run python tools/cache_pure.py --src <raw PURE> --dest D:/pure_zarr

Alignment: PURE's oximeter runs at ~60 Hz against ~30 fps video, so the
waveform is resampled to one sample per frame. ``--align timestamp`` (default)
interpolates on the real nanosecond timestamps both streams carry;
``--align index`` reproduces the legacy toolbox's index-only ``resample_ppg``
for comparison against published numbers. The choice is recorded in the store's
``alignment`` root attr.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm

#: Written to every store's root attrs. The loader's admission gate refuses
#: anything below 1.0.0, the raw-frame cache format floor.
TOOL_VERSION = "1.0.0"

#: Frames per chunk along T, matching the Neckflix cache's (C, 32, H, W).
FRAMES_PER_CHUNK = 32

#: PURE's six recording setups, from the dataset ReadMe. Written as the
#: ``setup_name`` root attr so motion condition is a filterable attribute.
SETUP_NAMES = {
    "01": "steady",
    "02": "talking",
    "03": "slow_translation",
    "04": "fast_translation",
    "05": "small_rotation",
    "06": "medium_rotation",
}

#: PNG filenames carry the capture timestamp in nanoseconds, and the raw
#: directories also hold unrelated artefacts (.avi, _masks/, .npy), so the
#: frame list is built from this pattern rather than a bare *.png glob.
FRAME_PATTERN = re.compile(r"^Image(\d+)\.png$")

RECORDING_PATTERN = re.compile(r"^(\d{2})-(\d{2})$")


def find_frame_dir(recording_dir: Path) -> Path:
    """The directory holding the PNG sequence for one recording.

    The published dataset nests the images one level deeper under a directory
    of the same name (``01-01/01-01/Image*.png``); some redistributed copies
    flatten it (``01-01/Image*.png``). Both are accepted.
    """
    nested = recording_dir / recording_dir.name
    if nested.is_dir() and any(FRAME_PATTERN.match(p.name) for p in nested.iterdir()):
        return nested
    return recording_dir


def list_frames(frame_dir: Path):
    """``(paths, timestamps_ns)`` for one recording, in capture order.

    Sorted on the parsed integer timestamp rather than lexicographically: the
    two agree for PURE's fixed-width names, but only one of them says why.
    """
    found = []
    for path in frame_dir.iterdir():
        match = FRAME_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise FileNotFoundError(f"No Image*.png frames under {frame_dir}")
    found.sort()
    stamps = np.array([ts for ts, _ in found], dtype=np.int64)
    return [path for _, path in found], stamps


def read_sidecar(json_path: Path):
    """``(waveform, waveform_ns, image_ns)`` from a PURE ``{recording}.json``.

    ``/FullPackage`` is the oximeter stream (``Value.waveform`` per record) and
    ``/Image`` is the camera's own frame-timestamp list.
    """
    with json_path.open() as handle:
        sidecar = json.load(handle)
    try:
        packages = sidecar["/FullPackage"]
        images = sidecar["/Image"]
    except KeyError as err:
        raise ValueError(
            f"{json_path.name}: expected '/FullPackage' and '/Image' keys, "
            f"got {sorted(sidecar)}"
        ) from err
    waveform = np.array([rec["Value"]["waveform"] for rec in packages], dtype=np.float64)
    waveform_ns = np.array([rec["Timestamp"] for rec in packages], dtype=np.int64)
    image_ns = np.array([rec["Timestamp"] for rec in images], dtype=np.int64)
    return waveform, waveform_ns, image_ns


def align_trace(waveform, waveform_ns, image_ns, *, mode: str) -> np.ndarray:
    """Resample the oximeter waveform to one sample per frame.

    ``timestamp`` interpolates on the real capture times, which both streams
    record in nanoseconds. Frames outside the oximeter's span (PURE's two
    streams start and stop within ~10 ms of each other, so at most a sample or
    two at each end) take the nearest edge value, as ``np.interp`` clamps.

    ``index`` is the legacy toolbox's ``resample_ppg``: linear interpolation on
    a 1-based uniform grid that ignores both timestamp arrays. It assumes the
    two streams start and stop together, which for PURE is very nearly true --
    it is kept for fidelity against published results, not because it is more
    correct.
    """
    target = len(image_ns)
    if mode == "timestamp":
        order = np.argsort(waveform_ns)
        return np.interp(image_ns.astype(np.float64),
                         waveform_ns[order].astype(np.float64),
                         waveform[order])
    if mode == "index":
        count = len(waveform)
        return np.interp(np.linspace(1, count, target),
                         np.linspace(1, count, count), waveform)
    raise ValueError(f"Unknown alignment mode {mode!r}")


def measured_fps(stamps_ns: np.ndarray) -> float:
    """Frame rate implied by the capture timestamps.

    The rate is a property of the data, so it is measured rather than assumed
    to be the nameplate 30 Hz; the loader tolerates 1% of jitter around a
    nominal rate and only decimates across a genuinely different one.
    """
    if len(stamps_ns) < 2:
        raise ValueError("Need at least two frames to measure a frame rate")
    elapsed_s = (stamps_ns[-1] - stamps_ns[0]) / 1e9
    if elapsed_s <= 0:
        raise ValueError("Frame timestamps are not increasing")
    return (len(stamps_ns) - 1) / elapsed_s


def write_store(recording_dir: Path, dest_dir: Path, *, align: str,
                overwrite: bool, limit_frames: int = 0):
    """Write one ``{recording}.zarr`` store; return its path (None if skipped).

    ``complete: true`` is written only after every frame and the trace are on
    disk, so an interrupted run leaves a store the loader's admission gate
    rejects rather than a plausible-looking short one.
    """
    name = recording_dir.name
    match = RECORDING_PATTERN.match(name)
    if not match:
        raise ValueError(f"{name!r} is not a PURE 'SS-TT' recording name")
    subject, setup = match.groups()

    store_path = dest_dir / f"{name}.zarr"
    if store_path.exists():
        if not overwrite:
            print(f"  {name}: exists, skipping (use --overwrite to rewrite)")
            return None
        shutil.rmtree(store_path)

    json_path = recording_dir.parent / f"{name}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"{name}: missing sidecar {json_path}")

    frame_paths, frame_ns = list_frames(find_frame_dir(recording_dir))
    waveform, waveform_ns, image_ns = read_sidecar(json_path)

    # The PNG names and the sidecar's /Image list are two independent records
    # of the same capture times; a mismatch means the pair is not a pair.
    if len(frame_paths) != len(image_ns):
        raise ValueError(
            f"{name}: {len(frame_paths)} PNG frames but {len(image_ns)} "
            f"/Image records in {json_path.name} -- recording and sidecar disagree"
        )
    image_ns = np.sort(image_ns)
    if not np.array_equal(frame_ns, image_ns):
        raise ValueError(
            f"{name}: PNG filename timestamps do not match the sidecar's "
            "/Image timestamps"
        )

    if limit_frames:
        frame_paths = frame_paths[:limit_frames]
        frame_ns = frame_ns[:limit_frames]
        image_ns = image_ns[:limit_frames]

    fps = measured_fps(frame_ns)
    trace = align_trace(waveform, waveform_ns, image_ns, mode=align)

    with Image.open(frame_paths[0]) as probe:
        width, height = probe.size
    n_frames = len(frame_paths)

    root = zarr.open_group(str(store_path), mode="w")
    video = root.create_group("1").create_group("rgb").create_group("video")
    frames = video.create_array(
        "frames",
        shape=(3, n_frames, height, width),
        dtype="uint8",
        chunks=(3, FRAMES_PER_CHUNK, height, width),
    )

    buffer = np.empty((FRAMES_PER_CHUNK, height, width, 3), dtype=np.uint8)
    for start in tqdm(range(0, n_frames, FRAMES_PER_CHUNK),
                      desc=f"  {name}", unit="chunk", leave=False):
        stop = min(start + FRAMES_PER_CHUNK, n_frames)
        for offset, path in enumerate(frame_paths[start:stop]):
            with Image.open(path) as image:
                # PURE's PNGs are RGB already; convert() normalises anything
                # palette- or grayscale-encoded to the 3-channel contract.
                decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if decoded.shape[:2] != (height, width):
                raise ValueError(
                    f"{name}: {path.name} is {decoded.shape[1]}x{decoded.shape[0]}, "
                    f"expected {width}x{height} -- the cache holds one resolution"
                )
            buffer[offset] = decoded
        # (t, H, W, C) -> (C, t, H, W), the loader's frame layout.
        frames[:, start:stop] = np.transpose(buffer[: stop - start], (3, 0, 1, 2))

    video.create_array(
        "timestamps_us",
        data=((frame_ns - frame_ns[0]) // 1_000).astype(np.int64),
    )
    video.attrs.update({"fps": float(fps), "num_frames": int(n_frames)})

    # Canonical name, lowercase: config TRACES are canonicalised (BVP -> PPG)
    # before the loader looks the group up, so this must be 'ppg'.
    root["1"]["rgb"].create_group("ppg").create_array(
        "data", data=trace.astype(np.float64))

    root.attrs.update({
        "recording": name,
        # Zero-padded to three digits: normalise_participant() pads any id the
        # CLI or a config offers ('P01', '01', 1) to that width, and a LOSO
        # filter that does not match this attr exactly selects nothing.
        "participant": subject.zfill(3),
        "subject": subject,
        "setup": setup,
        "setup_name": SETUP_NAMES.get(setup, "unknown"),
        "dataset": "PURE",
        "source_resolution": [height, width],
        "resized_to": None,
        "alignment": align,
        "tool_version": TOOL_VERSION,
        "complete": True,
    })
    return store_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True,
                        help="raw PURE root (the directory of SS-TT folders)")
    parser.add_argument("--dest", type=Path, required=True,
                        help="cache directory to write *.zarr stores into")
    parser.add_argument("--recordings", nargs="+", default=None,
                        help="only these recordings, e.g. 01-01 02-01 (default: all)")
    parser.add_argument("--align", choices=("timestamp", "index"), default="timestamp",
                        help="waveform-to-frame alignment (default: timestamp)")
    parser.add_argument("--overwrite", action="store_true",
                        help="rewrite stores that already exist")
    parser.add_argument("--limit-frames", type=int, default=0,
                        help="cache at most N frames per recording (smoke runs)")
    args = parser.parse_args()

    if not args.src.is_dir():
        parser.error(f"--src {args.src} is not a directory")

    candidates = sorted(
        path for path in args.src.iterdir()
        if path.is_dir() and RECORDING_PATTERN.match(path.name)
    )
    if args.recordings:
        wanted = set(args.recordings)
        unknown = wanted - {path.name for path in candidates}
        if unknown:
            parser.error(f"--recordings not found under {args.src}: {sorted(unknown)}")
        candidates = [path for path in candidates if path.name in wanted]
    if not candidates:
        parser.error(f"No PURE 'SS-TT' recording directories under {args.src}")

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"PURE -> zarr: {len(candidates)} recording(s), {args.src} -> {args.dest}")
    print(f"alignment: {args.align}")

    written = []
    for recording_dir in candidates:
        store_path = write_store(
            recording_dir, args.dest,
            align=args.align, overwrite=args.overwrite,
            limit_frames=args.limit_frames,
        )
        if store_path is None:
            continue
        video = zarr.open_group(str(store_path), mode="r")["1"]["rgb"]["video"]
        print(f"  {store_path.name}: {video.attrs['num_frames']} frames @ "
              f"{video.attrs['fps']:.4f} fps")
        written.append(store_path)

    print(f"done: {len(written)} store(s) written to {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
