# PURE Cache Spec

PURE (Pulse Rate Detection Dataset): 10 subjects, 6 recording setups each,
about one minute per recording, captured as timestamped PNG sequences at
640x480 and 30 fps with a finger pulse oximeter (about 60 Hz) as the
reference. Dataset page:
<https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure>.
Cite: Stricker, R., Mueller, S., Gross, H.-M., "Non-contact Video-based
Pulse Rate Measurement on a Mobile Service Robot", Ro-Man 2014.

This spec is the mapping `dataset/cachers/pure/` implements, from the raw
release onto [the cache contract](../../docs/cache-contract.md). The
validator is the acceptance test; the local copy is `D:/PURE/PURE`.

## Raw layout (as found, 2026-09-03)

```text
{input_dir}/
|-- 01-01/                      one directory per recording, "SS-TT"
|   |-- 01-01/                  same-named directory of frames
|   |   |-- Image1392643993642815000.png   capture time, epoch ns, in the name
|   |   |-- ...                 (2026 frames here; 1920-2579 across the set)
|   |   `-- Thumbs.db           stray; matched out by the Image<ns>.png pattern
|   `-- 01-01.json              sidecar: "/Image" and "/FullPackage" record lists
|-- ...
|-- 06-02/                      empty in the public release; not a recording
`-- 10-06/
```

`SS` is the subject (`01`-`10`), `TT` the setup (`01` steady, `02`
talking, `03` slow translation, `04` fast translation, `05` small rotation,
`06` medium rotation). 59 recordings carry data.

The JSON sidecar:

- `/Image`: one record per frame, `Timestamp` in epoch nanoseconds. In
  every recording these equal the frame filenames' timestamps exactly; the
  cacher refuses a recording where they do not.
- `/FullPackage`: one record per oximeter sample at about 59.5 Hz (57.7 in
  `04-01`), `Timestamp` on the same clock, and `Value` with `waveform`,
  `pulseRate`, `o2saturation`, `signalStrength`, `barGraph` and five
  boolean flags (`beep`, `droppingo2Sat`, `probeError`, `searching`,
  `searchingToLong`). The oximeter starts and stops within about 40 ms of
  the camera.

Measured across the set: camera 29.85-30.0 fps; durations 64-86 s.

## What the cacher writes

```text
01-01.zarr
|-- attrs                                see "Root attributes"
`-- 1/                                   attrs: fps=30.0 (nominal)
    `-- rgb/
        |-- timestamps_us/data  (T,) int64   /Image capture times, epoch microseconds
        |-- video/data          (3,T,H,W) uint8  RGB, chronological; Delta + blosc-zstd
        `-- ppg/data            (T,) float64     attrs: units="arb"
```

- **Frames**: every `Image<ns>.png`, ordered by the timestamp in the name,
  decoded with OpenCV and converted BGR to RGB, so the `rgb` modality's
  planes are R, G, B as the contract's channel map expects. `--resize H W`
  is applied per frame with area interpolation; without it frames are the
  native 640x480. Chunks are `(3, 32, H, W)`.
- **Timestamps**: absolute epoch microseconds (`ns // 1000`), so the capture
  time survives in the store. Strictly increasing is checked.
- **`ppg`**: the oximeter `waveform`, linearly interpolated at the frame
  timestamps (interpolation in seconds relative to the first frame, for
  precision). A frame outside the oximeter's span gets NaN, never a clamped
  edge value: typically the first frame (the oximeter starts a few
  milliseconds late) and, when the oximeter stops early, a NaN tail that is
  trimmed with the frames it covers. The reader treats NaN label samples as
  absent (finite-only statistics, zeroed after normalisation). Units are
  `arb`: the oximeter's waveform has no physical unit.
- A recording is exactly one perspective with one modality; there is no
  other camera to align to.

### Root attributes

Written verbatim or measured from the raw data; nothing is translated.

```python
{
  "participant": "01",                 # the subject token as the dataset spells it
  "recording": "01-01",
  "subject": "01",
  "setup": "01",
  "setup_name": "steady",              # from the published setup list
  "capture_start_utc": "2014-02-17T13:33:13.642+00:00",
  "duration_s": 67.7,
  "frame_rate_measured": 29.911,
  "oximeter": {
    "num_samples": 4018,
    "sample_rate_measured": 59.34,
    "start_offset_ms": 3.9,            # oximeter start minus first frame
    "end_offset_ms": -1.3,             # oximeter end minus last frame
    "pulseRate":      {"min": 65.0, "max": 83.0, "mean": 70.125},   # bpm, the device's own estimate
    "o2saturation":   {"min": 95.0, "max": 97.0, "mean": 95.561},   # percent
    "signalStrength": {"min": 4.0,  "max": 5.0,  "mean": 4.708},
    "barGraph":       {"min": 0.0,  "max": 11.0, "mean": 4.537},
    "flag_counts": {"beep": 151, "droppingo2Sat": 0, "probeError": 0,
                    "searching": 0, "searchingToLong": 0},          # samples with the flag set
  },
  "alignment": "timestamp",
  "source_resolution": [480, 640],
  "resized_to": [128, 128],            # or null
  "tool_version": "0.1.0",
  "complete": {"modalities": ["rgb"], "perspectives": ["1"]},   # the cacher's own skip/rebuild record
}
```

The per-sample oximeter fields other than the waveform are kept as
recording-level summaries because the trace vocabulary has no place for a
pulse-rate or SpO2 series; the raw JSON remains the source if a per-sample
series is ever wanted.

Configs filter on these values exactly as written: `participant: ["01"]`,
`setup_name: [steady, talking]`, and so on.

## Building

```bash
uv run --project dataset/cachers/pure pure-preprocess \
    --input-dir D:/PURE/PURE --output-dir <cache dir> --resize 256 256
uv run python tools/validate_cache.py <cache dir>
```

About 80 s per recording at 128x128 from the local drive, dominated by PNG
decoding; two workers by default.

## History (the legacy loader, deleted)

The pre-overhaul `PURELoader.py` (git tag `pre-overhaul`) sorted frames
lexicographically, read `cv2.imread` output converted BGR to RGB, took
`/FullPackage[*].Value.waveform` in list order and put it on the frame grid
by *index* (`np.interp` over a 1-based uniform grid, ignoring every
timestamp), and encoded the recording as the int `101` and the subject as
the int `1`. None of that is reproduced: alignment is by timestamp and the
identifiers are the dataset's own strings. Face cropping, chunking, frame
normalisation and the percentage splits were consumer-side and are not the
cache's business.
