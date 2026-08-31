# PURE Cache Spec

PURE (Pulse Rate Detection Dataset) contains videos of 10 subjects across 6
recording setups (steady, talking, and four head-motion variants, per the
dataset's published description), captured as timestamped PNG image
sequences with a finger pulse oximeter providing the reference pulse
waveform. Dataset page:
<https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure>.
Cite: Stricker, R., Mueller, S., Gross, H.-M., "Non-contact Video-based
Pulse Rate Measurement on a Mobile Service Robot", Proc. 23rd IEEE Int.
Symposium on Robot and Human Interactive Communication (Ro-Man 2014),
Edinburgh, Scotland, UK, pp. 1056-1062, IEEE 2014. This spec describes how
to write `{recording}.zarr` stores for PURE conforming to the cache
contract in docs/architecture.md; it replaces the deleted legacy loader
(`PURELoader.py`, retrievable at the `pre-overhaul` git tag).

## Raw layout

```
data/PURE/
|-- 01-01/                  <- one directory per recording, "SS-TT"
|   |-- 01-01/              <- nested dir of same name: the image sequence
|   |   |-- Image1392643993642815000.png
|   |   |-- ...
|   |-- 01-01.json          <- sensor data (pulse waveform etc.)
|-- 01-02/
|   |-- 01-02/
|   |-- 01-02.json
|...
|-- ii-jj/
```

Discovery glob used by the loader: `data_path + os.sep + "*-*"` (every
directory whose name contains a hyphen). Inside each recording directory
the frames live in a nested directory with the *same name* as the
recording, and the sensor JSON is `{recording}.json` beside it. Per the
published dataset, PNG filenames encode the capture timestamp in
nanoseconds (e.g. `Image1392643993642815000.png`); the loader relied only
on their lexicographic order.

**Redistributed copies vary.** The local subset at
`C:\Users\20759193\source\repos\hr_detection\datasets\PURE` flattens the
nesting (`01-01/Image*.png`, no inner directory) and drops extra artefacts
beside the frames — `01-01.avi`, `01-01_masks/`, `01-01_points_labels_dict.npy`
— which are not part of PURE. `tools/cache_pure.py` therefore accepts both
layouts and selects frames by the `Image<digits>.png` pattern rather than a
bare `*.png` glob.

## Video

- Format: PNG image sequence (one file per frame), not a video container.
- Legacy read: `sorted(glob.glob(video_file + '*.png'))` over the nested
  image directory, then per file `cv2.imread(png_path)` followed by
  `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`. Result stacked to
  `(T, H, W, 3)`.
- Frame order: plain lexicographic sort of the full paths. Because the
  timestamp filenames are fixed-width, lexicographic order equals
  chronological order.
- Pixel format: uint8, 0-255, **RGB** after the explicit BGR->RGB
  conversion (cv2.imread returns BGR).
- Resolution: never asserted by the loader; whatever the PNGs decode to
  (the published dataset is 640x480).
- fps: never read from the raw data by the loader. Every repo config pins
  `FS: 30` for PURE (the dataset's published camera rate is 30 Hz);
  write `fps: 30.0` unless derived from the image timestamps.

### Motion-augmented variant (DATA_AUG 'Motion')

With `DATA_AUG: ['Motion']` the legacy loader instead read a single `.npy`
file found by `glob(os.path.join(recording_dir, recording_name, '*.npy'))`
(only the first match was loaded), produced by the external MA-rPPG Video
Toolbox. Decoding rules (inherited `read_npy_video`, replicate exactly):

- integer dtype with values in [0, 255]: cast each frame to uint8 and keep
  only the first 3 channels (`frame.astype(np.uint8)[..., :3]`);
- floating dtype with values in [0.0, 1.0]: `np.round(frame * 255)` cast
  to uint8, first 3 channels;
- anything else: hard error.

The `.npy` frames are already RGB — no BGR->RGB swap is applied on this
branch. If motion-augmented PURE is cached, it should become *separate*
stores (or a distinguishing root attr such as `augmentation: "motion"`),
never silently replace the real frames.

## Physiological traces

### ppg (zarr key: `ppg`)

**The key is `ppg`, not `bvp`.** `config.py` canonicalises
`INTERFACE.TRACES` through `neural_methods/signals.py` before the loader
sees them, and `BVP` is an alias of `PPG`; the loader then looks the trace
group up by the *lowercased canonical* name. A group written as `bvp` is
never found, and the sample is dropped for having no labels.

- Source: `{recording}.json`, parsed with `json.load`. The loader read
  exactly `labels["/FullPackage"]`, a list of records, taking
  `label["Value"]["waveform"]` from each record, in list order. Everything
  else in the JSON (per-record timestamps, other `Value` fields such as
  pulse rate and SpO2, and the `/Image` frame-timestamp list, per the
  published dataset) was ignored.
- Units: arbitrary units from the finger pulse oximeter; the loader
  applied no scaling. Store as float in these native units.
- Native sampling rate: not read by the loader (the published sensor rate
  is 60 Hz, vs 30 fps video, so there are roughly 2x as many waveform
  samples as frames).
- Frame alignment (the contract requires one sample per frame): the legacy
  loader used the inherited `resample_ppg`, which is a pure index-based
  linear interpolation ignoring all timestamps:

  ```python
  np.interp(np.linspace(1, N, target_length),
            np.linspace(1, N, N), input_signal)
  ```

  with `N = len(waveform)` and `target_length = frames.shape[0]` (the
  decoded frame count). Replicate this exactly for fidelity with legacy
  results. Note the ambiguity: the JSON carries real timestamps for both
  frames and waveform samples, so a timestamp-based alignment is possible
  and arguably more correct, but it is *not* what the legacy pipeline did.

  **Resolved:** `tools/cache_pure.py` defaults to the timestamp alignment
  (`np.interp` on the two nanosecond timestamp arrays) and keeps the legacy
  index grid behind `--align index`. Measured on the four-recording subset,
  the two streams start and stop within ~10 ms of each other, so the index
  grid is a good approximation — but only the timestamp path stays correct
  if a capture ever drops frames, and waveform-level phase accuracy is what
  this repo is ultimately validating. The mode used is recorded in the
  store's `alignment` root attr, so a cache built either way says so.

The JSON's pulse-rate and SpO2 series could become additional traces
(`hr`, `spo2`) in a future cache; the legacy loader never read them, so
this spec does not define their parsing.

## Identity and attributes

- Recording directory name is `SS-TT` (subject `SS`, setup/trial `TT`),
  e.g. `01-01` ... `10-06`.
- Legacy index: the directory name with the hyphen removed, cast to int
  (`01-01` -> 101, `10-06` -> 1006) — leading zero dropped. Cached chunks
  were named e.g. `101_input0.npy`.
- Participant parse (for LOSO): the first two characters of the hyphenless
  name, cast to int (`subject = int(subject_trail_val[0:2])`). The legacy
  BEGIN/END splitter grouped recordings by this subject number so splits
  never shared subjects.
- Proposed root attrs:
  - `recording`: the raw directory name, e.g. `"01-01"`
  - `participant`: the subject token, unprefixed and **zero-padded to three
    digits** — `"001"`, not `"01"`. `normalise_participant()` pads every id
    the CLI or a config offers (`P01`, `01`, `1`) to that width before
    filtering, so a two-digit attr here would make every LOSO fold select
    nothing, silently.
  - `setup` (or `trial`): the trial token, e.g. `"01"`. Per the dataset's
    published description the six setups are 01 steady, 02 talking,
    03 slow translation, 04 fast translation, 05 small rotation,
    06 medium rotation — useful as a motion/task filter.

## Store mapping (implemented)

Written by `tools/cache_pure.py`; read by `PUREDataset`
(`PURELoader.py`), whose channel map is `{"R": ("rgb", 0), "G": ("rgb", 1),
"B": ("rgb", 2)}` — so frames are stored in RGB order.

```text
01-01.zarr
  attrs: complete: true, tool_version: "1.0.0", dataset: "PURE",
         recording: "01-01", participant: "001", subject: "01",
         setup: "01", setup_name: "steady", alignment: "timestamp",
         source_resolution: [480, 640], resized_to: null
  1/                          <- single perspective
    rgb/
      video/frames            (3, T, H, W) uint8, RGB, chunked (3, 32, H, W)
      video/timestamps_us     (T,) int64, relative to the first frame
      video/  attrs: num_frames (= T, required), fps (measured, ~29.9-30.0)
      ppg/data                (T,) float64, oximeter a.u., one sample per
                              frame (see the alignment note above)
```

`fps` is *measured* from the frame timestamps rather than pinned to the
nameplate 30, matching what the Neckflix preprocessor writes; the loader
treats anything within 1% as the same nominal rate.

`complete: true` is written only after the frames and the trace are on
disk, so an interrupted run leaves a store the admission gate rejects
rather than a plausible-looking short one.

Build it with:

```bash
uv run python tools/cache_pure.py --src <raw PURE> --dest <cache dir>
```

## Quirks

- The image directory is *nested with the same name* as the recording
  directory; the JSON sits one level up. The loader built the glob prefix
  as `os.path.join(path, filename, "")` (note the trailing separator).
- The recording "id" was an int, so `01-01` and a hypothetical `1-01`
  would collide as 101. Keep the raw hyphenated string as the store name
  to avoid this.
- `resample_ppg` interpolates over `np.linspace(1, N, ...)` — a 1-based
  uniform index grid, not timestamps. All timestamp data in the JSON is
  ignored by the legacy pipeline (see ambiguity note above).
- The waveform is roughly 2x oversampled relative to frames (60 Hz vs
  30 fps published rates); linear interpolation *downsamples* it to frame
  rate with no anti-alias filtering. Replicating this is a fidelity
  choice, not a signal-processing recommendation.
- Motion-augmented `.npy` branch: only the first glob match is read; float
  inputs are assumed to be in [0, 1]; channels beyond the first 3 are
  dropped (see Video section for exact rules).
- `USE_PSUEDO_PPG_LABEL` generated POS-based pseudo labels from the frames
  instead of reading the JSON (POS signal, 2nd-order Butterworth bandpass
  0.70-3 Hz via filtfilt, Hilbert-envelope normalisation). This is a
  training-time label substitution derived from the video, not raw data —
  it does not belong in the cache.
- No missing-file tolerance: absent PNGs simply shorten the video; an
  absent JSON raised. No subject exclusions existed for PURE.

## Not replicated

Face cropping (Haar Cascade / YOLO5Face), spatial resizing, chunking,
`DiffNormalized` / `Standardized` frame and label normalisation, and the
BEGIN/END percentage splits with their CSV file lists are all
consumer-side or obsolete in the new pipeline. The cache stores raw
full-length RGB frames and the physical-unit trace only; `DATA_TYPE`
transforms and resizing happen in `neural_methods/frame_transforms.py`,
and splits are root-attr filters at load time.
