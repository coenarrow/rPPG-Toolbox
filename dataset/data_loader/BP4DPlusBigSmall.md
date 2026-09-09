# BP4D+ BigSmall Cache Spec

The BigSmall variant of BP4D+ is not a different raw dataset — it is the AU
(facial action unit) subset of BP4D+ used by the BigSmall multi-task model
(pulse + respiration regression, multilabel AU classification), adapted from
https://github.com/girishvn/BigSmall (Narayanswamy et al., "BigSmall:
Efficient Multi-Task Learning for Disparate Spatial and Temporal
Physiological Measurements", arXiv:2303.11573). Cite the BP4D+ publications
listed in [BP4DPlus.md](BP4DPlus.md) plus the BigSmall paper. This spec
describes how to write `{recording}.zarr` stores for BP4D+ BigSmall
conforming to the cache contract in docs/architecture.md; it replaces the
deleted legacy loader (`BP4DPlusBigSmallLoader.py`, retrievable at the
`pre-overhaul` git tag).

**Recommendation (design choice): do not build a separate cache.** The raw
data is BP4D+; everything BigSmall adds — AU traces, fold membership, the
task subset — fits into the BP4DPlus stores as extra trace groups and root
attrs. This spec therefore documents only what *differs* from
[BP4DPlus.md](BP4DPlus.md); the raw layout, video decode, the eight
physiological files, and their units are identical and specified there.

## Raw layout

As in [BP4DPlus.md](BP4DPlus.md). Additionally read here:

```
RawData/AUCoding/
|-- AU_OCC/
|   |-- F001_T1.csv                   binary occurrence coding, one per
|   |-- ... {subj}_{task}.csv         AU-coded recording
|-- AU_INT/
|   |-- AU06/
|   |   |-- F001_T1_AU06.csv          intensity coding (0-5), one folder
|   |-- AU10/ AU12/ AU14/ AU17/       per intensity-coded AU
```

Plus the fold lists tracked in this repo (referenced by the legacy configs'
`FOLD.FOLD_PATH`):

```
dataset/BP4D_BigSmall_Subject_Splits/
  Split1_Train_Subjects.csv   94 subjects     Split1_Test_Subjects.csv   46
  Split2_Train_Subjects.csv   93 subjects     Split2_Test_Subjects.csv   47
  Split3_Train_Subjects.csv   93 subjects     Split3_Test_Subjects.csv   47
```

Each CSV has one column `subjects` with 4-char codes (`F006`, `M013`, ...).
Train + Test = 140 unique subjects per split, and the three Test sets are
pairwise disjoint and jointly cover all 140 (verified against the CSVs), so
every subject belongs to the Test set of exactly one split.

## Video

Same zips and decode as [BP4DPlus.md](BP4DPlus.md) (RGB uint8, archive-order
iteration, `ValueError('EMPTY VIDEO')` on no match). Differences, none of
which the cache replicates:

- Decode-time downsample went to `BIGSMALL.RESIZE.BIG_H x BIG_W` (config.py
  defaults `144 x 144`) via `downsample_frame`: when `dim_h == dim_w` it
  first took a square crop `frame[int((frame.shape[0]-frame.shape[1])):,:,:]`
  (keeps the bottom `width` rows — assumes portrait frames; for landscape
  frames the negative index silently produces a non-square crop), then
  `cv2.resize(..., interpolation=cv2.INTER_AREA)`. Store native-resolution
  frames instead.
- The dual-resolution "big" (144x144, `BIG_DATA_TYPE`) / "small" (9x9,
  `SMALL_DATA_TYPE`) pathways, including the second `crop_face_resize` to
  `SMALL_W x SMALL_H = 9 x 9`, are model-input preprocessing — consumer-side
  in the new world (`frame_transforms.py` + model-carried resize).
- Latent bug, flagged for the record: both `crop_face_resize` calls in the
  loader's `preprocess` pass 8 positional args where the current
  `BaseLoader.crop_face_resize` takes 9 (the `backend` parameter was added
  after this loader was written), so the DO_PREPROCESS path would raise
  `TypeError` as committed.

## Physiological traces

Same eight files, units, and proposed keys as [BP4DPlus.md](BP4DPlus.md).
This loader read all eight (`read_raw_phys_labels`), including
`EDA_microsiemens.txt` (the filename authority for the EDA ambiguity noted
there), via the same header-consuming `pd.read_csv(...).to_numpy().flatten()`.
Its alignment differs slightly from the plain loader — quoted exactly:

```python
len_Xsub = data_dict['X'].shape[0]   # frame count
bp_wave = np.interp(np.linspace(0, len(bp_wave), len_Xsub),
                    np.arange(0, len(bp_wave)), bp_wave)
# identically for HR_bpm, resp_wave, resp_bpm, mean_BP, sys_BP, dia_BP, eda
```

(0-based endpoints vs `resample_ppg`'s 1-based; the query point at
`len(bp_wave)` lies past the last sample index and `np.interp` clamps it to
the final value.) Same design choice as the plain spec: pure linear
interpolation, no anti-aliasing — pick one behavior for the cache-writer and
record it.

### AU labels

Source files, per AU-coded recording `{subj}` `{task}` (e.g. `F001`, `T1`):

- Occurrence: `AUCoding/AU_OCC/{subj}_{task}.csv`, read with `header=0` (the
  file has a real header row). Column 0 is the **1-indexed video frame
  number**; columns 1..34 are the AU occurrence codes, assumed by the loader
  to be ordered exactly as:

  ```
  AU_num = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 39]                      # 34 AUs, key "AU01".."AU39"
  ```

- Intensity (0-5 per the loader comment), for five AUs only:
  `AU_int_num = [6, 10, 12, 14, 17]`, from
  `AUCoding/AU_INT/AU{nn}/{subj}_{task}_AU{nn}.csv`, read with
  `header=None`; column 0 is the frame number, column 1 the intensity. The
  loader asserted the row count equals the AU_OCC row count. Keys
  `"AU06int"`, `"AU10int"`, `"AU12int"`, `"AU14int"`, `"AU17int"` — 39 AU
  series total.

**Frame alignment**: AU rows cover a contiguous 1-indexed frame span
`[start_frame, end_frame]` taken from the first and last rows of AU_OCC
(`AUs[0,0]`, `AUs[-1,0]`). The loader front-padded with `-1` for frames
before `start_frame` and back-padded with `-1` up to the video frame count,
then **cropped every array (video included) to the coded span**
`[start_frame-1 : end_frame-1]` inclusive (0-indexed). The shape check
raising `ValueError('Shape Mismatch')` enforced that the padded AU length
equals the frame count, which implies contiguous rows — `F041T7` is excluded
precisely because its video and AU lengths disagree.

**AU value semantics (ambiguity)**: the loader's "outlier" step implies the
occurrence CSVs contain values other than 0/1 (BP4D+ documents 9 as
"unknown/occluded" — verify in the user guide). The legacy cleanup line

```python
au[np.where(au != 0) and np.where(au != 1)] = 0
```

is a Python `and` of two index tuples, which evaluates to the second
operand — so it actually zeroes *every* value != 1, including intensity
values 2-5 in the same array slice. Faithful replication would destroy the
intensity coding; the cache must store the **raw integer codes** and leave
any cleanup consumer-side.

**Mapping AUs onto the cache contract (design choice, not from the loader)**:
AU labels are per-frame integer classification labels, not continuous
traces. Proposal: store each AU series as its own lowercase trace group
(`au01`, `au02`, ..., `au39`, and `au06int`, `au10int`, `au12int`, `au14int`,
`au17int`) as `(T,)` float64 index-aligned to the *full-length* frames, with
the raw codes as float values and `NaN` (not `-1`) outside the coded span —
the zarr loader already NaN-pads trace tails, and `label_mask` handles
recordings that lack a trace, so recordings without AU coding simply omit
the `au*` groups rather than carrying `-1` fill. Consequences to accept
explicitly: per-window `zscore`/`minmax` label normalisation is meaningless
for integer class codes, and the trainer in `src/trainer.py` has no classification
head — a consumer wanting BigSmall-style AU training must treat `au*` keys
specially. The legacy AU-span video crop becomes consumer-side (recoverable
as the non-NaN span of any `au*` trace).

## Identity and attributes

As [BP4DPlus.md](BP4DPlus.md) (recording `F008T8`, participant `F008`, sex,
task), with these differences:

- **Task subset**: only `T1`, `T6`, `T7`, `T8` were enumerated ("only ones
  that have AU labels"). In a unified cache this is a filter, not a schema
  difference; record `au_coded: true/false` per store (design choice) so the
  subset is selectable without opening trace groups.
- **Exclusion**: `F041T7` ("data sample has mismatch length for video frames
  and AU labels") — write no AU traces for it (or no store, matching the
  loader; design choice — this spec recommends writing the store without
  `au*` groups and letting filters drop it).
- The loader's internal `subject = int(index[1:4])` dropped the sex letter,
  colliding F and M subjects with equal numbers; its split grouping used the
  4-char code so no harm resulted. Use the 4-char code as `participant`.
- **3-fold membership**: the legacy pipeline filtered subjects through
  `FOLD.FOLD_PATH` CSVs (column `subjects`, matched against `index[0:4]`)
  named in the configs `BP4D_BP4D_BIGSMALL_FOLD{1,2,3}.yaml` (the legacy
  config tree was deleted in the Phase 5 close-out; git history has the
  files). Proposal
  (design choice): derive a root attr `bigsmall_fold` in {1,2,3} — the split
  whose `Test` CSV contains the subject — so the three cross-validation
  folds become attribute filters. The six CSVs under
  `dataset/BP4D_BigSmall_Subject_Splits/` are the membership authority;
  preserve them (or their content) alongside the cache, since the fold
  assignment is not derivable from the raw data.

Root attrs: those of BP4DPlus.md plus `au_coded` and `bigsmall_fold`.

## Proposed store mapping

The [BP4DPlus.md](BP4DPlus.md) store, extended for AU-coded recordings:

```
F008T8.zarr
  attrs: complete: true, tool_version: ">=1.0.0", recording: "F008T8",
         participant: "F008", sex: "F", task: "T8",
         au_coded: true, bigsmall_fold: 2
  1/
    rgb/
      video/frames  (3, T, H, W) uint8       video/ attrs: num_frames, fps: 25
      abp/ systolic_bp/ diastolic_bp/ mean_bp/ hr/ resp/ rr/ eda/
                    (T,) float64, physical units (see BP4DPlus.md)
      au01/data ... au39/data                (T,) float64 raw integer codes,
      au06int/ au10int/ au12int/ au14int/    NaN outside the AU-coded span
      au17int/
```

## Quirks

- Everything in BP4DPlus.md's Quirks applies (zip order, header-consumed
  first sample, F042T11 — though T11 never enters here since only
  T1/T6/T7/T8 are enumerated).
- Legacy label munging a naive cache-writer must NOT bake in (all
  consumer-side; cache stores raw values): systolic clamped to [5, 250],
  diastolic to [5, 200], EDA to [1, 40]; the AU `!= 1 -> 0` zeroing (buggy,
  see above); `LABEL_TYPE` DiffNormalized/Standardized applied only to
  `bp_wave`, `resp_wave`, `pos_bvp`, `pos_env_norm_bvp`.
- The legacy output was a flat `(T, 49)` label array in this exact column
  order — recorded here because it is the only place the label vocabulary
  was enumerated: `[bp_wave, HR_bpm, systolic_bp, diastolic_bp, mean_bp,
  resp_wave, resp_bpm, eda, AU01, AU02, AU04, AU05, AU06, AU06int, AU07,
  AU09, AU10, AU10int, AU11, AU12, AU12int, AU13, AU14, AU14int, AU15,
  AU16, AU17, AU17int, AU18, AU19, AU20, AU22, AU23, AU24, AU27, AU28,
  AU29, AU30, AU31, AU32, AU33, AU34, AU35, AU36, AU37, AU38, AU39,
  pos_bvp, pos_env_norm_bvp]`, missing labels `-1`-filled. The named trace
  groups replace it.
- POS pseudo-labels here differ from BaseLoader's: the Butterworth band was
  HR-adaptive around the trial-mean `Pulse Rate_BPM` —
  `hr_freq = mean(HR_bpm)/60`, `halfband = 20 / fs` (with `fs = FS = 25`;
  the code comment claims "+/- 20 bpm" but `20/fs` is 0.8 Hz = 48 BPM — the
  code is authoritative), band clamped to [0.70, 3] Hz, 2nd-order
  Butterworth `filtfilt`, then Hilbert envelope normalisation; both
  `pos_bvp` and `pos_env_norm_bvp` were kept. Derived from video, so not
  cached.
- Frames were saved as pickled dicts `{0: big_clip, 1: small_clip}` plus a
  `.npy` label per chunk — an artifact of the dual-resolution npy pipeline,
  entirely replaced by the store.
- `read_raw_phys_labels` returns `None` on `FileNotFoundError` (printing
  "Label File Not Found At Basepath"), which would crash the caller — in
  practice every enumerated recording had all eight files; a cache-writer
  should fail loudly on a missing physiology file.

## Not replicated

The big/small dual-resolution preprocessing (144x144 / 9x9,
`BIG_DATA_TYPE`/`SMALL_DATA_TYPE`), face cropping, decode-time downsampling,
frame/label DiffNormalized/Standardized transforms, chunking, `BEGIN`/`END`
splits, fold-filtered file-list CSVs, the AU-span video crop, label clamps,
and POS pseudo-labels are consumer-side or derived and are absent from the
cache. The store holds raw full-length frames, physical-unit traces, and raw
AU codes only; fold membership survives as a root attr.
