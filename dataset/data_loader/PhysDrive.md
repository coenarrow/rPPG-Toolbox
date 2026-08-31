# PhysDrive Cache Spec

PhysDrive is a multimodal remote physiological measurement dataset for
in-vehicle driver monitoring: aligned RGB frame sequences of drivers with
synchronized BVP, ECG, respiration, and SpO2 ground truth, collected on real
roads. See https://github.com/WJULYW/PhysDrive-Dataset; cite Jiyao Wang,
Xiao Yang, Qingyong Hu, Jiankai Tang, Can Liu, Dengbo He, Yuntao Wang,
Ying-Cong Chen, Kaishun Wu (2025), "PhysDrive: A Multimodal Remote
Physiological Measurement Dataset for In-vehicle Driver Monitoring". This
spec describes how to write `{recording}.zarr` stores for PhysDrive
conforming to the cache contract in docs/architecture.md; it replaces the
deleted legacy loader (`PhysDriveLoader.py`, retrievable at the
`pre-overhaul` git tag).

## Raw layout

`DATA_PATH` pointed at the `On-Road-rPPG/` directory (the shipped config
used `/home/jywang/Data/On-Road-rPPG`):

```
On-Road-rPPG/
|-- AFH1/                        one directory per subject (4-char code)
|   |-- A1/                      one directory per session
|   |   |-- Align/
|   |   |   |-- *.png            aligned RGB frames, one file per frame
|   |   |-- Label/
|   |   |   |-- BVP.mat
|   |   |   |-- ECG.mat
|   |   |   |-- RESP.mat
|   |   |   |-- SPO2.mat
|   |-- A2/ B1/ B2/ C1/ C2/      further sessions, same structure
|-- AFH2/
|-- ...
|-- CMZ2/
|-- processed/                   skipped by the loader if present
```

Enumeration: `glob(DATA_PATH/*)` for subjects (skipping a directory whose
basename is exactly `"processed"`), then `glob(subject_dir/*)` for sessions
— every subdirectory counts; no name validation. The loader's docstring and
the upstream README both show sessions `A1, A2, B1, B2, C1, C2`.

Optionally present per session (only under `DATA_AUG: ['Motion']`):
`{session_dir}/*.npy` motion-augmented frame arrays — see Quirks.

## Video

- Source: `Align/*.png`, collected with
  `sorted(glob.glob(os.path.join(video_file, "*.png")))` — **lexicographic
  sort**. Correct temporal order therefore relies on zero-padded frame
  filenames (ambiguity: the loader assumed it; a cache-writer should sort
  numerically on the frame index in the filename after confirming the
  naming pattern against the raw data).
- Decode: `cv2.imread(f)` then `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` —
  store RGB channel order, uint8, 0-255, at native resolution (the
  resolution is not recorded in the code; read it from the pngs).
- fps: not read from any file. The shipped config
  (`PhysDrive_PhysDrive_PhysDrive_TSCAN_BASIC.yaml`, in the legacy config
  tree deleted in the Phase 5 close-out — git history has it)
  set `FS: 30` in all data blocks, and the pseudo-label branch used
  `fs=self.config_data.FS`. Write `fps: 30` (verify against the PhysDrive
  documentation).
- The loader raised on an empty/missing `Align/` directory
  (`NotADirectoryError` / "Empty frames").

## Physiological traces

Four MATLAB `.mat` files under `Label/` per session. The legacy loader read
**only BVP**; ECG/RESP/SPO2 exist in the layout (docstring + README) but
were never parsed, so all facts below marked unverified must be confirmed
against the raw files or the PhysDrive repository before writing.

| Raw file | Signal | mat key | Units | Native rate | Proposed key |
| --- | --- | --- | --- | --- | --- |
| `BVP.mat` | blood volume pulse | `"BVP"` | unknown (a.u.) | unknown | `bvp` |
| `ECG.mat` | electrocardiogram | unverified (likely `"ECG"`) | unknown | unknown | `ecg` |
| `RESP.mat` | respiration | unverified (likely `"RESP"`) | unknown | unknown | `resp` |
| `SPO2.mat` | pulse oximetry | unverified (likely `"SPO2"`) | unknown (likely %) | unknown | `spo2` |

BVP parse — the only one in code, quoted exactly:

```python
waves = sio.loadmat(bvp_file)["BVP"].flatten()
```

**Units and native sampling rates are not encoded anywhere in the loader**
(ambiguity): it aligned BVP purely by length ratio. Consult the PhysDrive
dataset documentation for the sensor sampling rates and units, and record
them in store attrs.

**Alignment to frames** — the contract requires `(T,)` traces index-aligned
to the frames, so this becomes the cache-writer's job. The legacy loader
did, for BVP:

```python
target_length = frames.shape[0]
bvps = BaseLoader.resample_ppg(bvps, target_length)
# where resample_ppg is exactly:
np.interp(np.linspace(1, input_signal.shape[0], target_length),
          np.linspace(1, input_signal.shape[0], input_signal.shape[0]),
          input_signal)
```

then asserted `len(bvps) == target_length`. Pure linear interpolation over
sample index — no timestamps, no anti-alias filtering. Apply the same
treatment to ECG/RESP/SPO2 (design choice: the loader never handled them;
for ECG especially, naive linear decimation to 30 Hz destroys the QRS
morphology — flag whichever resampling the writer adopts in provenance
attrs, or reconsider whether ECG belongs at frame rate at all).

## Identity and attributes

- Recording id: `f"{subject_name}_{session_name}"`, e.g. `"AFH1_A1"` — the
  loader's `unique_id`, formed from the two directory basenames.
- Participant id (LOSO): the subject directory name, e.g. `"AFH1"`. The
  loader itself mapped subject names to sequential integers
  (`subject_id_map[subject_name] = len(subject_id_map) + 1`) purely for its
  `BEGIN`/`END` subject-disjoint split; the integers depend on filesystem
  glob order and must NOT be replicated — use the stable folder name.
- **Folder code scheme (ambiguity)**: the loader treats `AFH1` and `A1`
  entirely opaquely — it never decodes them. The PhysDrive publication
  assigns meaning to the subject-code letters and to the session letters
  (vehicle / subject / time-of-day / road-condition groupings), but none of
  that is in this repository's code. If vehicle/time/road root attrs are
  wanted for filtering, derive the decoding from the PhysDrive dataset
  documentation and record it in the cache-writer — do not guess it from
  this spec.

Root attrs: `recording` (e.g. `"AFH1_A1"`), `participant` (e.g. `"AFH1"`),
`session` (e.g. `"A1"`), plus the mandatory `complete: true` and
`tool_version`; optional decoded attrs (vehicle, time, road) pending the
documentation check above.

## Proposed store mapping

One store per subject x session, single camera perspective:

```
AFH1_A1.zarr
  attrs: complete: true, tool_version: ">=1.0.0", recording: "AFH1_A1",
         participant: "AFH1", session: "A1"
  1/
    rgb/
      video/frames   (3, T, H, W) uint8, RGB, native resolution
      video/  attrs: num_frames (required), fps: 30
      bvp/data       (T,) float64, index-aligned to frames
      ecg/data       (T,) float64
      resp/data      (T,) float64
      spo2/data      (T,) float64
```

## Quirks

- A top-level `processed/` directory is data-adjacent clutter, not a
  subject — skip it (the loader special-cased exactly the basename
  `"processed"`).
- **Signal-quality clip filtering**: after chunking, the legacy loader
  scored every BVP clip with neurokit2 — `ppg_peaks(..., sampling_rate=FS,
  method='elgendi')` then `ppg_quality(..., method='templatematch')`,
  `np.nanmean` over the per-sample quality — and discarded clips with
  quality `< 0.5` (also scoring 0 for signals shorter than 10 samples,
  constant signals, no detected peaks, or a `ValueError`). Whole sessions
  with zero surviving clips were dropped. This is training-set curation,
  not a property of the data: the cache stores everything (design choice,
  flagged — if BigSmall-style parity with published PhysDrive results is
  wanted, quality filtering must be reimplemented consumer-side; the
  constants above are the complete recipe).
- Per-session error swallowing: any exception in a session's preprocessing
  was caught and printed (`[Error] Failed to process ...`), silently
  shrinking the dataset. A cache-writer should fail loudly (or at minimum
  write no store and log) so missing stores are auditable.
- `DATA_AUG: ['Motion']` bypassed `Align/` and loaded `{session_dir}/*.npy`
  via `read_npy_video`: `np.load` of the **first** globbed file only,
  accepting integer arrays in [0, 255] (cast uint8) or float arrays in
  [0, 1] (scaled by 255, rounded, cast uint8), keeping the first 3
  channels. These are externally generated augmentation arrays, not raw
  data — they do not go in the cache.
- The integer `subject` ids were glob-order-dependent (see Identity) — the
  legacy `BEGIN`/`END` splits over them were therefore not reproducible
  across filesystems.
- Lexicographic png ordering (see Video).

## Not replicated

Face cropping (Haar cascade, `LARGE_BOX_COEF: 1.5` in the shipped config),
`DiffNormalized`/`Standardized` frame and label transforms, 180-frame
chunking, quality-based clip dropping, `BEGIN`/`END` percentage splits,
file-list CSVs, motion-augmentation `.npy` inputs, and POS pseudo-PPG
labels are consumer-side, curation, or derived, and are absent from the
cache. The store holds raw full-length RGB frames and physical-unit traces
only.
