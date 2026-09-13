# Evaluation

`scripts/eval.py DIR [DIR ...] [--reading-seconds 30]`

Each `DIR` is a records directory `scripts/infer.py` wrote, or a run
directory as shorthand for its `test_records/`. The evaluation reads
`meta.json` and the per-recording trace tables and nothing else: no
config file, no checkpoint. It is not yet torch-free, though — a known
limitation, not the design's goal: no file under `src/evaluation/`
mentions torch, but the signal registry it needs
(`src/signal_transforms.py`) shares a module with torch-dependent label
normalisation, reached transitively through `src/outputs.py` ->
`src/interface.py` -> `src/config.py`. Importing `src.evaluation` still
pulls in torch, `src.interface` and `src.config`, so a laptop with no
torch install cannot run it yet. Every `<recording>/<perspective>/`
folder under each `DIR` is scored afresh on every run, and its three
tables are written beside its trace tables. Clause and page numbers
below are the printed ones in `standards/`.

## The prediction being scored

The trace table's `mean` column: the average of every strided window
covering the frame. Its `std` is the repeatability across windows and
`label` the reference in physical units. Averaging overlapping windows
smooths the prediction; the per-window columns stay in the trace tables
for anyone who wants the unsmoothed one. Errors are prediction minus
reference everywhere, the sign every standard uses.

## Per recording and camera

Written beside the trace tables: `beats.csv`, `readings.csv`,
`rates.csv`. Nothing is cached; a run overwrites whatever an earlier one
left. The three files carry no dataset, participant, recording or
perspective columns: a folder's place in the records directory says
where it sits.

**Readings.** Non-overlapping stretches of `--reading-seconds` from the
first covered frame; a trailing remainder shorter than half a reading is
dropped. 30 s is the ISO 81060-2 invasive reference interval (clause 6.2.4
b), p. 21); about 10 s is the ISO 81060-3 device segment (clause 5.1.3,
p. 14, and A.2, p. 27); IEEE 1708 records 60 s (clause 4.4.2, p. 24).

**Beats** (`src/evaluation/beats.py`). One detector for every cardiac
trace: detrend and bandpass to 0.6 to 3.3 Hz, `find_peaks` with the
minimum beat distance from the top of the band and the prominence a
registry fraction of the cleaned range (`beat` in
`src/signal_transforms.py`), each candidate moved to the raw extremum
within a quarter of the median beat interval. A beat spans trough to
trough: `max` is its peak, `min` the lower trough, `mean` the area under
the curve over the duration, the MAP definition of ISO 81060-2 clause
6.2.4 e), p. 22. Predicted beats match the nearest reference beat within
40 percent of the median reference interval, closest first, one to one.

`beats.csv`: `signal`, `beat`, `t_ref`, `t_pred` (blank on a miss),
`ref_max`, `ref_mean`, `ref_min`, `pred_max`, `pred_mean`, `pred_min`
(blank for shape-class signals). One row per *reference* beat; a
predicted beat with no reference match has no row here at all, but is
counted in `readings.csv`'s `n_pred_beats`.

`readings.csv`, per reading and signal: `t_start`, `t_end`,
`reading_seconds` (the cut length that produced the table); `n_ref_beats`,
`n_pred_beats`, `n_matched` (recall is matched over reference, precision
matched over predicted); for absolute signals `ref_<s>_mean`,
`ref_<s>_sd`, `pred_<s>_mean`, `pred_<s>_sd` over the beats for `s` in
`max`, `mean`, `min` (systolic, MAP, diastolic for ABP; peak, mean, trough
for CVP; for a non-cardiac absolute signal the sample mean, SD, max and
min), `err_<s>` and `err_<s>_deadband` (ISO 81060-2 clause 6.2.5, p. 22:
zero inside the reference mean ± SD, else the distance to the nearer
limit); `waveform_mad`, `waveform_rmse`, `waveform_r`, `waveform_ccc`
over the reading's samples (IEEE 1708 equations (3) and (4), p. 28).

`rates.csv` (`src/evaluation/rate.py`), per reading and source: `ref_hr`,
`pred_hr`, `err_hr`, `snr`, `macc`. Each cardiac trace is cleaned
(smoothness-prior detrend, then a zero-phase first-order bandpass to
0.6 to 3.3 Hz), its plain periodogram taken, zero-padded to a power of
two, and the largest in-band bin read as the rate, on the label and on
the prediction. `snr` is the power within 6 bpm of the reference rate
and its second harmonic over the rest of the band, in dB; `macc` the
maximum amplitude of cross-correlation over every lag. When a reading
carries two or more cardiac traces two sources join them: `FUSED`, the
rate of the geometric mean of the traces' spectra, each normalised to
unit in-band power (so the frequency every trace agrees on wins, and a
peak only one trace has is suppressed), and `MEDIAN`, the median of the
per-trace rates. SNR and MACC are on the combined trace and are not
comparable with the upstream toolbox's per-window numbers.

## Not covered

Pooling recordings into per-participant or per-dataset summaries, the
clinical standards' criteria and coverage tables, figures and the
report: to be rebuilt on top of the three tables above. ESH 2023 and ISO
81060-1. Calibration and time-since-initialisation. CVP beats are
detected with the shared detector and are expected to be unreliable;
`readings.csv`'s `n_ref_beats`, `n_pred_beats` and `n_matched` say
whether they are.
