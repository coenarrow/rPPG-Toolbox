# UBFC-rPPG Cache Spec

UBFC-rPPG is a remote-photoplethysmography dataset of one facial video per
subject recorded with a low-cost webcam, with a synchronised
pulse-oximeter ground truth. Dataset page:
<https://sites.google.com/view/ybenezeth/ubfcrppg>. Cite: S. Bobbia,
R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue
segmentation for remote photoplethysmography", Pattern Recognition
Letters, 2017. This spec describes how to write `{recording}.zarr` stores
for UBFC-rPPG conforming to the cache contract in docs/architecture.md; it
replaces the deleted legacy loader (`UBFCrPPGLoader.py`, retrievable at
the `pre-overhaul` git tag).

## Raw layout

```
data/UBFC-rPPG/
|-- subject1/
|   |-- vid.avi
|   |-- ground_truth.txt
|-- subject2/
|   |-- vid.avi
|   |-- ground_truth.txt
|...
|-- subjectn/
|   |-- vid.avi
|   |-- ground_truth.txt
```

Discovery glob used by the loader: `data_path + os.sep + "subject*"`.
This is the "DATASET_2" (realistic) release format; the loader had no
support for the older "DATASET_1" release (which uses a `gtdump.xmp`
ground-truth format) — only `vid.avi` + `ground_truth.txt` directories
were readable.

## Video

- Container: `vid.avi`, decoded with `cv2.VideoCapture(video_file)`. The
  loader called `VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)` and then read
  frames in a `VidObj.read()` loop until failure — i.e. every frame the
  codec yields, in order.
- Each frame: `cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)`. Result
  stacked to `(T, H, W, 3)`.
- Pixel format: uint8, 0-255, **RGB** after the explicit BGR->RGB
  conversion.
- Resolution: never asserted by the loader; whatever the AVI decodes to
  (the published dataset is 640x480, uncompressed 8-bit RGB).
- fps: never read from the container by the loader
  (`CAP_PROP_FPS` was not queried). Every repo config pins `FS: 30` for
  UBFC-rPPG (the dataset's published rate is ~30 Hz); write `fps: 30.0`
  unless derived from the container.

### Motion-augmented variant (DATA_AUG 'Motion')

With `DATA_AUG: ['Motion']` the loader instead read a single `.npy` file
found by `glob(os.path.join(subject_dir, '*.npy'))` (only the first match
was loaded), produced by the external MA-rPPG Video Toolbox. Decoding
rules (inherited `read_npy_video`, replicate exactly):

- integer dtype with values in [0, 255]: cast each frame to uint8 and keep
  only the first 3 channels (`frame.astype(np.uint8)[..., :3]`);
- floating dtype with values in [0.0, 1.0]: `np.round(frame * 255)` cast
  to uint8, first 3 channels;
- anything else: hard error.

No BGR->RGB swap is applied on this branch (the `.npy` frames are already
RGB). If cached, augmented recordings should be separate stores or carry a
distinguishing root attr (e.g. `augmentation: "motion"`).

## Physiological traces

### bvp (zarr key: `bvp`)

- Source: `ground_truth.txt` in the subject directory. The loader read the
  whole file, split on `"\n"`, and parsed **only the first line**:
  `bvp = [float(x) for x in str1[0].split()]` — whitespace-separated
  floats. The remaining lines were ignored (per the dataset description
  they carry the heart-rate series and per-sample timestamps; the loader
  never verified this).
- Units: arbitrary units from the pulse oximeter; no scaling applied.
  Store as float in these native units.
- Frame alignment: **none**. Unlike PURE and UBFC-PHYS, the loader applied
  no `resample_ppg` — the first line was used as-is, one value per frame,
  on the assumption that the PPG series and the frame count already match
  1:1. The legacy code never checked the lengths; a cache writer must make
  the assumption explicit: assert `len(bvp) == num_frames`, and treat a
  mismatch as a decision the legacy code does not cover (flag it rather
  than silently resampling, since resampling would diverge from legacy
  behaviour).

## Identity and attributes

- One recording per subject directory, named `subjectN` (the published
  DATASET_2 numbering is not contiguous).
- Legacy index: `re.search('subject(\d+)', data_dir).group(0)` — note
  `group(0)`, so the index is the *full* `subjectN` string, prefix
  included. Cached chunks were named e.g. `subject1_input0.npy`.
- Participant parse (for LOSO): the digits, i.e. `group(1)` of the same
  pattern. The legacy BEGIN/END splitter did not group by subject (one
  recording per subject makes that moot) — it took an index range over the
  raw glob order, which is OS-dependent and unsorted (see Quirks).
- Proposed root attrs:
  - `recording`: the directory name, e.g. `"subject1"`
  - `participant`: the numeric token, unprefixed, e.g. `"1"`

There is no task/condition structure in the raw naming — one video per
subject is all the dataset encodes.

## Proposed store mapping

One store per subject: `{cache_dir}/subject1.zarr`.

```
subject1.zarr
  attrs: complete: true, tool_version: ">=1.0.0",
         recording: "subject1", participant: "1"
  1/                          <- single perspective
    rgb/
      video/frames            (3, T, H, W) uint8, RGB channel order
      video/  attrs: num_frames (= T, required), fps (30.0)
      bvp/data                (T,) float, oximeter a.u., taken verbatim
                              from line 1 of ground_truth.txt
```

Channel map consumers will use `{"R": ("rgb", 0), "G": ("rgb", 1),
"B": ("rgb", 2)}`, so store RGB order (i.e. keep the loader's BGR->RGB
conversion).

## Quirks

- `group(0)` vs `group(1)`: the legacy recording id kept the `subject`
  prefix; participant filtering needs the bare digits. Keep both facts
  straight when writing `recording` and `participant` attrs.
- The BVP series is used **without resampling** — the only one of the
  three UBFC/PURE-family loaders to do so. Do not "helpfully" resample.
- Only line 1 of `ground_truth.txt` is data; lines 2+ exist but were never
  parsed. If HR (reportedly line 2) is ever wanted as a trace, that is new
  parsing outside this spec's fidelity guarantee.
- Frame count comes from decoding, not metadata: the loader counted
  whatever `cv2.VideoCapture.read()` yielded. Different OpenCV/FFmpeg
  builds can decode a different number of frames from the same AVI; the
  cache writer should record the decoded count as `num_frames` and align
  the (unresampled) trace to it by assertion, not arithmetic.
- Glob order was never sorted before the BEGIN/END split, so legacy
  percentage splits were OS-dependent. Irrelevant to the cache (splits are
  now attr filters), but explains historical irreproducibility.
- `USE_PSUEDO_PPG_LABEL` generated POS-based pseudo labels from the frames
  (POS signal, 2nd-order Butterworth bandpass 0.70-3 Hz via filtfilt,
  Hilbert-envelope normalisation) instead of reading `ground_truth.txt`.
  Training-time substitution, derived from video — not cached.
- No missing-file tolerance and no subject exclusions: an absent
  `vid.avi` or `ground_truth.txt` raised.

## Not replicated

Face cropping (Haar Cascade / YOLO5Face), spatial resizing, chunking,
`DiffNormalized` / `Standardized` frame and label normalisation, and the
BEGIN/END percentage splits with their CSV file lists are all
consumer-side or obsolete in the new pipeline. The cache stores raw
full-length RGB frames and the physical-unit trace only; `DATA_TYPE`
transforms and resizing happen in `src/frame_transforms.py`,
and splits are root-attr filters at load time.
