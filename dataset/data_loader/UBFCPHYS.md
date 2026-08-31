# UBFC-PHYS Cache Spec

UBFC-PHYS is a multimodal database for psychophysiological studies of
social stress: each subject is recorded in three tasks (per the dataset's
published description, T1 rest, T2 speech, T3 arithmetic), with wristband
BVP as the physiological reference. Dataset page:
<https://sites.google.com/view/ybenezeth/ubfc-phys>. Cite: R. Meziati
Sabour, Y. Benezeth, P. De Oliveira, J. Chappe, F. Yang, "UBFC-Phys: A
Multimodal Database For Psychophysiological Studies Of Social Stress",
IEEE Transactions on Affective Computing, 2021. This spec describes how to
write `{recording}.zarr` stores for UBFC-PHYS conforming to the cache
contract in docs/architecture.md; it replaces the deleted legacy loader
(`UBFCPHYSLoader.py`, retrievable at the `pre-overhaul` git tag).

## Raw layout

```
RawData/
|-- s1/
|   |-- vid_s1_T1.avi
|   |-- vid_s1_T2.avi
|   |-- vid_s1_T3.avi
|   |...
|   |-- bvp_s1_T1.csv
|   |-- bvp_s1_T2.csv
|   |-- bvp_s1_T3.csv
|-- s2/
|   |-- vid_s2_T1.avi
|   |...
|-- sn/
```

Discovery glob used by the loader:
`data_path + os.sep + "s*" + os.sep + "*.avi"` — one *recording per video
file* (subject x task), not per subject directory. The matching BVP file
is `bvp_{index}.csv` in the same directory, where `index` is parsed from
the video filename (see Identity). The raw release also contains other
sidecar files (e.g. EDA CSVs and subject info files, per the dataset
description); the legacy loader never read them.

## Video

- Container: `vid_s{n}_T{k}.avi`, decoded with
  `cv2.VideoCapture(video_file)`. The loader called
  `VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)` and then read frames in a
  `VidObj.read()` loop until failure — every frame the codec yields, in
  order.
- Each frame: `cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)`. Result
  stacked to `(T, H, W, 3)`.
- Pixel format: uint8, 0-255, **RGB** after the explicit BGR->RGB
  conversion.
- Resolution: never asserted by the loader; whatever the AVI decodes to
  (the published dataset is 1024x1024).
- fps: never read from the container by the loader. Every repo config pins
  `FS: 35` for UBFC-PHYS (the dataset's published rate is 35 Hz); write
  `fps: 35.0` unless derived from the container.

Unlike the PURE and UBFC-rPPG loaders, this loader had **no
`DATA_AUG: ['Motion']` / `.npy` branch** — it always read the AVI, and any
DATA_AUG setting was ignored by this loader (motion-augmented UBFC-PHYS is
mentioned upstream but was not wired into this loader's read path).

## Physiological traces

### bvp (zarr key: `bvp`)

- Source: `bvp_{index}.csv` beside the video (e.g. `bvp_s1_T1.csv`),
  parsed with `csv.reader`; the loader appended `float(row[0])` for
  **every** row — first column only, no header skipped (the raw files are
  headerless single-column CSVs; if a header were present the parse would
  raise on `float(...)`).
- Units: arbitrary units from the wristband PPG (an Empatica E4 per the
  dataset description; native rate 64 Hz per the same description — the
  loader read neither fact from the data). No scaling applied; store as
  float in native units.
- Frame alignment (the contract requires one sample per frame):
  `bvps = BaseLoader.resample_ppg(bvps, frames.shape[0])`, a pure
  index-based linear interpolation ignoring timestamps:

  ```python
  np.interp(np.linspace(1, N, target_length),
            np.linspace(1, N, N), input_signal)
  ```

  with `N = len(bvp_csv_rows)` and `target_length` the decoded frame
  count. This maps the full BVP series onto the full video duration by
  index ratio alone — it assumes both start and stop together. Replicate
  exactly for fidelity (see Quirks for the aliasing caveat).

The EDA sidecar files could become an additional `eda` trace in a future
cache; the legacy loader never read them, so this spec does not define
their parsing.

## Identity and attributes

- One recording per video file. Legacy index:
  `re.search('vid_(.*).avi', video_path).group(1)` — e.g. `"s1_T1"`.
  Cached chunks were named e.g. `s1_T1_input0.npy`.
- Participant parse (for LOSO): the `s{n}` token before the underscore
  (the legacy exclusion filter split it off with `rsplit('_', 1)[0]` on
  chunk names). Store the number itself as the attr, e.g. `"1"`.
- Task: the `T{k}` token after the underscore — `T1` / `T2` / `T3`.
- The legacy BEGIN/END splitter took an index range over the raw glob
  order (unsorted, OS-dependent) and did *not* group by subject, so a
  percentage split could put the same subject's tasks in different
  splits. Obsolete now; splits are attr filters.
- Proposed root attrs:
  - `recording`: the parsed index, e.g. `"s1_T1"`
  - `participant`: unprefixed number, e.g. `"1"`
  - `task`: e.g. `"T1"` (rest), `"T2"` (speech), `"T3"` (arithmetic)

## Proposed store mapping

One store per subject-task video: `{cache_dir}/s1_T1.zarr`.

```
s1_T1.zarr
  attrs: complete: true, tool_version: ">=1.0.0",
         recording: "s1_T1", participant: "1", task: "T1"
  1/                          <- single perspective
    rgb/
      video/frames            (3, T, H, W) uint8, RGB channel order
      video/  attrs: num_frames (= T, required), fps (35.0)
      bvp/data                (T,) float, wristband PPG a.u.,
                              index-aligned to frames via the
                              resample_ppg mechanism
```

Channel map consumers will use `{"R": ("rgb", 0), "G": ("rgb", 1),
"B": ("rgb", 2)}`, so store RGB order (i.e. keep the loader's BGR->RGB
conversion).

## Quirks

- The BVP is heavily oversampled relative to frames (64 Hz vs 35 fps
  published rates); `resample_ppg`'s plain linear interpolation
  downsamples with no anti-alias filtering, and it aligns by index ratio,
  not by clock time — any lead/lag between wristband and camera start is
  silently stretched away. Replicating this is a fidelity choice, not a
  signal-processing recommendation.
- Frame count comes from decoding, not metadata; different OpenCV/FFmpeg
  builds may yield different counts for the same AVI. Record the decoded
  count as `num_frames` and resample the trace to that count.
- **Known-bad recordings.** The legacy pipeline filtered at *load* time
  via `FILTERING.USE_EXCLUSION_LIST` (an override of
  `load_preprocessed_data` matched chunk names against the list after
  stripping the `_input{i}.npy` suffix). The exclusion list used across
  this repo's UBFC-PHYS configs (e.g.
  `PURE_UBFC-PHYS_TSCAN_BASIC.yaml`) — preserved here because those
  configs were deleted in the Phase 5 close-out — is:

  ```
  s3_T1, s8_T1, s9_T1, s26_T1, s28_T1, s30_T1, s31_T1, s32_T1,
  s33_T1, s40_T1, s52_T1, s53_T1, s54_T1, s56_T1, s1_T2, s4_T2,
  s6_T2, s8_T2, s9_T2, s11_T2, s12_T2, s13_T2, s14_T2, s19_T2,
  s21_T2, s22_T2, s25_T2, s26_T2, s27_T2, s28_T2, s31_T2, s32_T2,
  s33_T2, s35_T2, s38_T2, s39_T2, s41_T2, s42_T2, s45_T2, s47_T2,
  s48_T2, s52_T2, s53_T2, s55_T2, s5_T3, s8_T3, s9_T3, s10_T3,
  s13_T3, s14_T3, s17_T3, s22_T3, s25_T3, s26_T3, s28_T3, s30_T3,
  s32_T3, s33_T3, s35_T3, s37_T3, s40_T3, s47_T3, s48_T3, s49_T3,
  s50_T3, s52_T3, s53_T3
  ```

  The configs give no reason for these exclusions (upstream describes the
  list generically as videos to exclude). Cache *all* recordings and keep
  exclusion a consumer-side filter (an `excluded: true` root attr, or an
  exclude list in experiment config) — baking it into the cache would
  discard data irreversibly.
- Task selection (`FILTERING.SELECT_TASKS` + `TASK_LIST`, matched by
  substring against the chunk name) was the other load-time filter; in the
  new pipeline it becomes an include filter on the `task` root attr.
- `USE_PSUEDO_PPG_LABEL` generated POS-based pseudo labels from the frames
  (POS signal, 2nd-order Butterworth bandpass 0.70-3 Hz via filtfilt,
  Hilbert-envelope normalisation) instead of reading the CSV.
  Training-time substitution, derived from video — not cached.
- No missing-file tolerance: an absent BVP CSV for a discovered video
  raised.

## Not replicated

Face cropping (Haar Cascade / YOLO5Face), spatial resizing, chunking,
`DiffNormalized` / `Standardized` frame and label normalisation, the
BEGIN/END percentage splits with their CSV file lists, and the load-time
exclusion/task filtering are all consumer-side or obsolete in the new
pipeline. The cache stores raw full-length RGB frames and the
physical-unit trace only; `DATA_TYPE` transforms and resizing happen in
`neural_methods/frame_transforms.py`, and splits/exclusions are root-attr
filters at load time.
