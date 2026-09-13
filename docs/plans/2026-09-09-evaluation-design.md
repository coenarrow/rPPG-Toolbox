# Evaluation: from window scores to standards-grade reports

**Goal:** replace the per-window evaluation with a three-layer package that
scores what `scripts/infer.py` actually writes, computes the statistics the
clinical blood-pressure standards ask for, and produces one PDF report per
evaluation. `scripts/eval.py` becomes the third real script of the chain.

**Why now:** nothing writes `test_records.pt` any more. `Trainer.test`
returns records in memory and `scripts/infer.py` writes them as the CSV
directory `src/outputs.py` describes, so `src/evaluation/records.py` loads a
file that no longer exists and README step 3 is broken.

**Standards consulted:** IEEE 1708-2025, ISO 81060-2:2018 with amendments 1
and 2, ISO 81060-3:2022 (all under `standards/`). ESH 2023 is not in the
folder and is not covered here. ISO 81060-1 covers non-automated cuff
devices and was skipped. Clause and page numbers below are the printed ones
and are the citations to put in code comments.

## Constraints

Binding, copied from `CLAUDE.md`:

1. Everything shared is written once. One beat detector, one reading
   segmenter, one set of agreement statistics, one plot set, used for every
   signal and every grouping. A second implementation is a bug in the first.
2. Legacy code is deleted, not adapted. No shims around the old `.pt` path.
3. Tests are a cost. One chain smoke test plus at most two dozen-line unit
   tests of pure functions. Stale tests are deleted in the same change.
4. Dependencies go through `uv add`, never pip.
5. No torch anywhere in `src/evaluation/`. The package reads CSV and JSON
   only, so a laptop can evaluate what the cluster inferred.

## What the evaluation reads

Per participant, `scripts/infer.py` writes (see `src/outputs.py`):

```text
<records dir>/                       default RUN_DIR/test_records/
  meta.json                          fs, window_frames, stride_frames,
                                     channels, traces, label_preprocessing,
                                     dataset, participant, run_dir, command,
                                     git, n_windows
  windows.csv                        one row per window: position and masks
  <recording>/<perspective>/<TRACE>.csv
                                     frame, t, label, mean, std, n, w<start>...
```

The trace table's `mean` column is the prediction the evaluation scores: the
average of every strided window covering the frame. `std` and `n` are its
repeatability. `label` is the reference in physical units, blank where the
trace is padded or uncovered. **Nothing else is read.** The per-window
columns exist for the ribbon plot's source data only and are never scored,
because per-window Bland-Altman is not useful to us and the standards score
readings, not windows.

## Package layout

```text
src/evaluation/
  __init__.py       package docstring
  beats.py          the shared beat detector, pairing, per-beat levels
  rate.py           spectral heart rate, fusion, median (exists; per reading now)
  post_process.py   upstream detrend / bandpass / SNR / MACC helpers (exists)
  recording.py      layer one: one recording and camera -> beats, readings, rates
  aggregate.py      layer two: many records dirs -> summary, criteria, coverage
  plots.py          layer three: every seaborn figure
  report.py         layer three: the PdfPages report and the text digest
scripts/eval.py     the entry point
```

Deleted: `src/evaluation/evaluate.py`, `src/evaluation/records.py`,
`tests/test_evaluate.py`. The agreement statistics `evaluate.py` defines
(`pearson`, `ccc`) move to `recording.py`, which needs them for the
waveform columns; `aggregate.py` imports them from there. The constants
`plots.py` imports from `evaluate.py` move to `aggregate.py`.

## Layer one: `recording.py` and `beats.py`

**Unit of work:** one `(recording, perspective)` folder. `recording.py`
exposes one function, `score_recording(folder, meta, reading_seconds)`, that
reads the folder's trace tables and writes three files beside them:

```text
<recording>/<perspective>/
  beats.csv       one row per reference beat per cardiac signal
  readings.csv    one row per reading per signal
  rates.csv       one row per reading per heart-rate source
```

All three carry no dataset, participant, recording or perspective columns.
Layer two adds those from the folder path and `meta.json` when it globs
them, so layer one never needs to know where it sits.

### Readings

A reading is one blood-pressure determination: a non-overlapping stretch of
the mean trace of `reading_seconds`, cut from the first covered frame. A
trailing stretch shorter than half the length is dropped. The length is a
flag on the script, default 30 s:

- ISO 81060-2 clause 6.2.4 b), p. 21: the invasive reference reading is the
  mean of the beat-by-beat values over at least 30 s.
- ISO 81060-3 clause 5.1.3 a) 1), p. 14: the segment matches the device's
  minimum output period; the rationale (A.2, p. 27) names 5 s to 10 s and a
  10-beat segment as typical. Set `--reading-seconds 10` for that style.
- IEEE 1708 clause 4.4.2, p. 24: three 60 s recordings per test.

`readings.csv` columns:

| column | meaning |
| --- | --- |
| `signal` | the trace |
| `reading` | index within the recording, 0-based |
| `t_start`, `t_end` | seconds at the interface rate |
| `n_ref_beats`, `n_pred_beats`, `n_matched` | detection counts, blank for non-cardiac signals |
| `ref_<s>_mean`, `ref_<s>_sd`, `pred_<s>_mean`, `pred_<s>_sd` for `s` in `max`, `mean`, `min` | mean and SD over the reading's beats of the per-beat level; absolute-class signals only |
| `err_<s>` | `pred_<s>_mean - ref_<s>_mean`, prediction minus reference, the sign every standard uses |
| `err_<s>_deadband` | ISO 81060-2 clause 6.2.5, p. 22: zero inside `ref_<s>_mean ± ref_<s>_sd`, otherwise the distance to the nearer limit |
| `waveform_mad`, `waveform_rmse`, `waveform_r`, `waveform_ccc` | per-sample agreement of `mean` against `label` over the reading; IEEE 1708 equations (3) and (4), p. 28, define the first and third |

`max`, `mean`, `min` are the registry's generic names. Reports render them
through `beat_labels()` so ABP reads systolic, MAP, diastolic and CVP reads
peak, mean, trough. For non-cardiac absolute signals (SpO2) the level
columns are the mean, SD, max and min of the samples over the reading, with
`n_*_beats` blank. Shape-class signals get the detection counts and the
waveform columns only.

The per-beat `mean` is the area under the curve over the beat divided by
its duration, which is the MAP definition in ISO 81060-2 clause 6.2.4 e),
p. 22, and ISO 81060-3 clause 3.5, p. 3.

A reading with no predicted beats carries blank predicted levels and stays in the table,
so the detection counts explain the blank.

### Beats

`beats.py` is the one detector, used on the label and on the prediction of
every cardiac signal. Algorithm:

1. Clean the trace as `rate.clean` does (detrend, zero-phase bandpass to the
   heart-rate band, 0.6 to 3.3 Hz).
2. `scipy.signal.find_peaks` on the cleaned trace times the signal's
   polarity, with `distance` from the top of the band (60 / 198 bpm at the
   rate) and `prominence` as the registry's fraction of the cleaned trace's
   peak-to-peak range.
3. Refine each candidate to the extremum of the raw trace within a quarter
   of the median inter-beat interval, so the systolic value is read off the
   real waveform, not the filtered one.
4. The beat spans trough to trough, the troughs being the minima of the
   polarity-signed trace between neighbouring peaks. Over that span the
   raw maximum is the systolic value, the raw minimum the diastolic, and
   the mean the MAP.

Pairing: predicted beats are matched one-to-one to the nearest reference
beat within 40 percent of the median reference inter-beat interval, greedy
by distance. Unmatched reference beats are misses; unmatched predicted beats
are spurious. Recall and precision per reading follow from the counts.

`beats.csv` columns: `signal`, `beat`, `t_ref`, `t_pred` (blank when
missed), `ref_max`, `ref_mean`, `ref_min`, `pred_max`, `pred_mean`,
`pred_min` (levels blank for shape-class signals). Spurious predicted beats
are not rows; they are counted in `readings.csv`.

Registry change in `src/signal_transforms.py`: every cardiac entry gains a
`beat` mapping, `{"polarity": +1 | -1, "prominence": <fraction>}`, read
through one accessor `beat_config(sig)`. Starting values: PPG, ABP, ECG
positive with prominence 0.3; CVP positive with prominence 0.2. CVP has no
systole and its a, c and v waves will be detected inconsistently. That is
not special-cased: the recall and precision columns and the beat overlay
plot are how we find out whether CVP beats are usable.

### Heart rate

`rate.py` keeps its estimator and fusion but its unit becomes the reading:
`window_rates(record, fs)` becomes `reading_rates(traces, fs)` taking the
`{signal: (label, prediction)}` slices of one reading. `rates.csv` columns
are unchanged: `reading`, `source`, `ref_hr`, `pred_hr`, `err_hr`, `snr`,
`macc`. The mean trace is smoother than any single window, so SNR and MACC
are not comparable with the upstream toolbox's per-window numbers; the
digest says so once.

## Layer two: `aggregate.py`

**Input:** one or more records directories. For each, `meta.json` and every
`<recording>/<perspective>/{beats,readings,rates}.csv`, tagged with
`dataset`, `participant`, `recording`, `perspective` and the reading's
absolute `t_start`. Layer one is re-run when a folder lacks its three files
or when its `readings.csv` was cut at a different length than the flag asks
(`t_end - t_start` of the first row); otherwise its files are trusted, so
pooling a LOSO sweep is reading CSVs.

**Output** into the evaluation directory:

| file | rows |
| --- | --- |
| `readings.csv`, `beats.csv`, `rates.csv` | the pooled, tagged tables |
| `summary.csv` | `group_by`, `group`, `signal`, `statistic`, `metric`, `value`, `n` |
| `criteria.csv` | `standard`, `clause`, `signal`, `measurand`, `group_by`, `group`, `metric`, `value`, `limit`, `pass` |
| `coverage.csv` | `standard`, `clause`, `signal`, `measurand`, `band`, `share`, `required`, `pass` |
| `changes.csv` | one row per change event: `participant`, `recording`, `perspective`, `signal`, `measurand`, `t_start`, `t_end`, `delta_ref`, `delta_pred`, `e_percent` |

**Groupings** (`group_by`): `all`, `dataset`, `participant`, `recording`
(recording and perspective together), `bp_category`. The blood-pressure
category is the participant's, from the mean reference systolic and
diastolic over all their readings, binned by IEEE 1708 Table 3, p. 22
(normal, elevated, stage 1, stage 2; the higher category wins).

**Summary metrics** per group, signal and statistic (`max`, `mean`, `min`,
`waveform`, `hr`): `bias`, `sd`, `loa_low`, `loa_high`, `mad`, `mapd`,
`rmse`, `cp5`, `cp10`, `cp15`, `pearson`, `ccc`, `n`. `hr` rows also carry
`snr` and `macc` as today (MAPD is `mapd` for levels and heart rate alike). `mad` is the mean absolute difference, IEEE
equation (1), p. 28; `cp<L>` is the share of errors within L mmHg, IEEE
clause 4.6.2, p. 32.

**Criteria**, each a row per signal and measurand, and per group where the
standard splits them:

- ISO 81060-2 criterion 1, clause 5.2.4.1.2 a), p. 9-10, and clause 6.2.6,
  p. 22 for the invasive route: pooled bias within ±5.0 mmHg, SD at most
  8.0 mmHg. Computed twice, on `err_<s>` and on `err_<s>_deadband`, labelled
  by clause.
- ISO 81060-2 criterion 2, clause 5.2.4.1.2 b), p. 10-11: the SD of the
  per-participant mean errors, centred on the **pooled** mean, against
  Table 1 (a constant in code, indexed by |pooled mean| to 0.1 mmHg). The
  standard exempts the invasive route from it; the row is still reported
  and labelled.
- ISO 81060-3 clause 5.1.4, p. 14-16: pooled bias within ±6.0 mmHg,
  corrected SD `s_corr` at most 10.0 mmHg, effective independent count
  `N_ind` at least 278. `s_corr`, the intra-class correlation and `N_ind`
  come from formulas (5), (6), (9) to (12) in their general unequal-count
  form. Reported as Type A on `err_<s>`, and as Type T (clause 5.2.4 b),
  p. 19: `s_corr` at most 6.0 mmHg, no bias limit) after subtracting each
  participant's offset from their first reading period.
- ISO 81060-3 clause 5.3.5, p. 22, change tracking: within each recording
  and perspective, every pair of readings whose start times differ by at
  most `--change-seconds` (default 60) is a candidate; it is a change event
  when either the reference delta or the predicted delta reaches 15 mmHg
  systolic, 10 diastolic or 12 MAP (clause 5.3.2, p. 20). Each event's
  error is `|delta_pred - delta_ref| / max(|delta_pred|, |delta_ref|)` in
  percent (formula (18)). Per participant the 50th and 85th percentiles;
  their averages over participants must be at most 25 and 50 percent.
- IEEE 1708 grades, clause 4.5.3.1 Table 6, p. 30: A for MAD ≤ 5, B for
  MAD in (5, 6] with |bias| ≤ 5, C for MAD in (6, 7] with |bias| ≤ 5, else
  D. Per measurand and per blood-pressure category (clause 4.5.3.4, p. 31:
  MAD ≤ 6 in every category but stage 2), and the overall grade is the worst
  cell. The waveform grade is Table 7, p. 30, from `waveform_mad` and
  `waveform_r`.

Counts the standards fix that we cannot, subjects, readings per subject,
equal readings per subject, appear in `criteria.csv` as achieved values with
`pass` blank.

**Coverage** (`coverage.csv`): the share of reference readings in each band,
against ISO 81060-2 clause 6.1.5, p. 18-19 (10 percent at each of SBP ≤ 100,
SBP ≥ 160, DBP ≤ 70, DBP ≥ 85) and ISO 81060-3 clause 4.3.3, p. 7-8 (5, 20,
20, 20, 5 percent bands on SBP, DBP and MAP). Plus the IEEE Table 5, p. 27,
histogram of blood-pressure change from baseline, the baseline being each
participant's first reading since we are calibration-free (IEEE clause
4.4.2, p. 24).

## Layer three: `plots.py` and `report.py`

Seaborn on matplotlib's Agg backend, every figure drawn by one function that
takes the tables and returns a figure. Titles use `beat_labels()` and
`signal_unit()`.

Aggregate figures, per signal and measurand, per grouping where marked:

- Bland-Altman with bias and limits, coloured by group, for `participant`
  and `recording`; the ISO amendment 2 form (difference against the mean of
  the two, clause 5.1.4 i), p. 3).
- Identity scatter, prediction against reference, per group.
- Error against blood-pressure change from baseline (IEEE clause 4.6.3,
  p. 35, figure 5).
- Histograms: reference readings and reference deltas (ISO 81060-3 clauses
  5.3.4 c) and d), p. 22), blood-pressure changes from baseline (IEEE
  figure 6), change-event error (ISO 81060-3 figure A.6).
- Reference reading against time since the recording start (ISO 81060-3
  clause 5.1.3 f), p. 14).
- Heart rate Bland-Altman per source, as today.

Per-recording figures, for a sample of recordings evenly spaced across the
pooled set (`--report-recordings`, default 4):

- The ribbon waveform: `mean ± std` of the prediction over the label for a
  few seconds, the repeatability plot you asked for.
- The beat overlay: label and prediction with detected beats marked, matched
  pairs joined, misses and spurious beats flagged, over the same stretch.
- The per-beat scatter of predicted against reference systolic and
  diastolic for the whole recording.
- The trend: reference and predicted MAP readings over time (ISO 81060-3
  figure A.3, p. 30), where lagged or under-reported changes show.

`report.py` writes `report.pdf` through `PdfPages`: a title page with what
was evaluated (directories, participants, readings, reading length, the
git state from `meta.json`), a page with `criteria.csv` and the coverage
table rendered as tables, the summary per grouping, then the figures. The
aggregate figures are also saved as individual PDFs so they can go into a
paper; the per-recording ones live only in the report. `digest.txt` is the
text twin of the first two pages and is printed to the console.

## `scripts/eval.py`

```text
uv run python scripts/eval.py runs/PHYSNET_PURE.01_<YYYYMMDDHHMM>
uv run python scripts/eval.py runs/A/test_records runs/B/test_records --out sweeps/loso
```

Positional `DIR [DIR ...]`, each a records directory or a run directory as
shorthand for its `test_records/`. Flags: `--out` (default
`<records dir>/evaluation/`, required when pooling), `--reading-seconds`
(30), `--change-seconds` (60), `--report-recordings` (4). No config, no
interface, no runtime, no torch. Layer one runs on every recording folder
that lacks its three files, then layers two and three write the evaluation
directory. The script exits non-zero on an empty pool with the offending
directory named.

## Dependencies, docs, tests

- `uv add seaborn` moves it from the dev group to a runtime dependency.
- `docs/evaluation.md`, new: every file and column above, every criterion
  with its clause, and the two flags. README step 3 and the output table,
  and `docs/adding_a_model.md` lines 404 and 428 and its output table, point
  at `scripts/eval.py`. `docs/plans/2026-09-08-model-migrations.md` line 97
  drops `test_records.pt`.
- Tests: `tests/test_evaluate.py` is deleted; its `to_physical` test moves
  to a new `tests/test_to_physical.py` unchanged. The chain test in
  `tests/test_scripts.py` gains the eval step over the infer output and
  asserts `readings.csv`, `criteria.csv` and `report.pdf` exist and the
  reading count matches the fixture. Two dozen-line unit tests: the beat
  detector on a synthetic sine (beat count and level), and the ISO 81060-2
  Table 1 lookup at the standard's own example (|mean| 4.2 mmHg gives 5.49).

## Risks and non-goals

- CVP beats: expected to be unreliable; measured, not fixed, in this change.
- The mean trace as the prediction averages up to `window / stride` windows,
  which smooths it. That is a design decision, stated in the docs; the
  per-window predictions remain in the trace tables for anyone who wants
  the unsmoothed version.
- ESH 2023 and ISO 81060-1 are not covered. ISO 81060-3's Scope page is
  missing from the scan; nothing here depends on it.
- No calibration and no time-since-initialisation analysis beyond the
  baseline-first-reading convention: the model is calibration-free.
