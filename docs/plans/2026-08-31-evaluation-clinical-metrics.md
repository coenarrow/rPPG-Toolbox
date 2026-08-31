# Evaluation & Clinical Metrics — design

Roadmap slot: **Phase 7** of
[the overhaul roadmap](2026-08-31-overhaul-roadmap.md). Status: design agreed
2026-08-31; implementation plan follows this document.

Phase 7's first instruction is "design doc first" — map IEEE 1708 /
ISO 81060 / ESH onto computable metrics, and be explicit about which criteria
are computable from our data versus which are study-design requirements a
metrics report can note but never satisfy. This is that document.

## 1. Why the current evaluation has to be rebuilt, not extended

The evaluation path is shaped around one question — *what heart rate did this
model predict* — because that was the whole of the upstream toolbox's job.
Three symptoms:

- `report_hr_metrics` is the only report there is. A signal is evaluated by
  deriving an HR from it and scoring that HR. For ABP and CVP the clinically
  interesting quantity is a pressure in mmHg, and the frequency content is the
  *least* interesting part.
- The one pressure-aware thing that exists, `plot_absolute_agreement`, works
  at window level with `np.max` / `np.min` over the whole window, and labels
  those panels "systolic" and "diastolic" for **every** absolute-class signal.
  A window maximum is not a systolic pressure, and CVP has no systole.
- The physical-unit inversion — the one operation every consumer needs — is
  written out three separate times: `MultiSignalTrainer._to_physical`,
  `metrics_report.to_physical`, and
  `tools/summarise_neckflix_outputs._to_physical`.

The prototypes parked in `evaluation/prototypes/` (roadmap Phase 1) already
reach past all of this: beat-level pressure statistics, HAC standard errors
that respect autocorrelation, moving-block bootstrap, Bland-Altman aggregated
per recording section. They are notebook scratch, not a module. Phase 7 step 3
is to consume them and delete them.

## 2. Scope

**In scope.** A full rebuild of the live evaluation path: `evaluation/` gains
a layered package, the trainer and the unsupervised predictor move onto it,
`tools/summarise_neckflix_outputs.py` becomes a thin CLI over the same
library, and the prototypes are consumed and deleted.

**Out of scope, and deliberately so.**

- `evaluation/metrics.py` and `evaluation/bigsmall_multitask_metrics.py` are
  **not touched**. They are imported by the seven legacy trainers and pinned
  by `tests/test_legacy_contract.py`, which deliberately guards that path.
  They die in Phase 6 with the last legacy trainer, not here.
- Change-tracking metrics (within-subject pressure change across postures) and
  a "predict this subject's mean" null baseline. Considered and declined.
  Recorded here so a later reader knows it was a decision, not an oversight.
- ISO 81060-2 criteria 1 and 2. Declined: they assume cuff determinations
  rather than a continuous waveform. The criteria layer is declarative, so
  adding them later is rows, not code.
- `PerSignalLoss`. See the observation in §16 — the same absolute-signal
  assumption sits in the training objective, but changing a loss is not an
  evaluation change.

## 3. Decisions

| # | Decision | Why |
| --- | --- | --- |
| D1 | Full rebuild of `evaluation/`, not an additive Phase 7 | The HR-centric shape of `report_hr_metrics` is the thing in the way; adding a clinical family beside it leaves two report styles |
| D2 | Measurements form a hierarchy: beat → window → section → recording → participant → cohort | The standards specify different levels; hardcoding one loses the others |
| D3 | Cohort level is computed by `tools/summarise_neckflix_outputs.py`, grown to accept a sweep directory | A LOSO fold has exactly one test participant, so no single run can produce a cohort band. One library, two callers — not a new pipeline stage |
| D4 | Criteria implemented: **ISO 81060-3** (beat-to-beat, invasive reference) and **IEEE 1708** (A–D MAE grading) | 81060-3 is the closest fit to an arterial-line reference; 1708 grading is the number cuffless-BP papers quote |
| D5 | Beats are **reference-anchored**; independent detection is reported separately as detection quality | Every reference beat yields exactly one pair, so agreement carries no selection bias. Matching-based pairing lets a bad prediction produce few pairs and therefore flattering error statistics |
| D6 | Standards' own aggregation governs clinical numbers; HAC / bootstrap standard errors apply to the DL-side metrics only | Layering our own uncertainty estimate on top of a criterion that defines its own aggregation reads as deviating from the standard |
| D7 | `TEST.METRICS` deleted; `TEST.REPORT` gates only bootstrap and figures | What applies to a signal follows from its class, so a metric can never go missing because a config forgot to ask. Net one fewer key, matching Phase 5's direction |
| D8 | Per-beat statistics are a uniform `max` / `mean` / `min` for every absolute signal; **display labels** come from the signal registry | Explicitly chosen: reuse one code path, and CVP's mean — the quantity that matters — is present. Per-signal labels stop the report printing "systolic CVP" |
| D9 | Thresholds live in one constants module with clause references and a printed provenance line | These numbers are written from general knowledge of the standards, not from their text. Nothing should print a clinical pass/fail without saying where the threshold came from |

## 4. Architecture

One path, two entry points. The trainer and the sweep reporter call identical
functions and differ only in which levels of the hierarchy they can populate.

```text
window records ──► records.py ──► beats.py ──► levels.py ──► metrics/ ──► report.py ──► digest + CSV/JSON
  (in-memory                                                     │                        plots.py
   or *.pickle)                                                  └── uncertainty.py
```

```text
evaluation/
  records.py         # WindowRecord; load from memory or from one-or-many pickles;
                     # the single physical-unit inversion
  beats.py           # cardiac clock, beat table, detection quality
  levels.py          # the hierarchy: grouping keys per level
  uncertainty.py     # HAC standard error, moving-block bootstrap (from prototypes)
  metrics/
    __init__.py      # family registry: which families apply to which signal class
    waveform.py      # MAE, RMSE, Pearson, CCC, MACC
    rate.py          # HR (FFT / peak), SNR — absorbs report_hr_metrics
    clinical.py      # per-beat agreement; generic criteria evaluator
    standards.py     # thresholds, grade bands, clause references, provenance
  report.py          # assemble the tidy frame; digest; CSV / JSON
  plots.py           # every figure (replaces BlandAltmanPy.py)

  post_process.py                # UNCHANGED — the DSP layer; unsupervised_methods depends on it
  metrics.py                     # UNTOUCHED — dies in Phase 6 with the legacy trainers
  bigsmall_multitask_metrics.py  # UNTOUCHED — same
```

### The tidy frame is the interface between layers

Every metric family appends rows to one long DataFrame. The digest, the CSV,
the JSON and every plot are views over it — which is what makes the trainer
and the sweep reporter the same code rather than two implementations that
drift apart.

| Column | Meaning |
| --- | --- |
| `level` | `beat` / `window` / `section` / `recording` / `participant` / `cohort` |
| `unit_id` | Identifier of the unit at that level |
| `signal` | Canonical signal name |
| `metric` | e.g. `mae`, `pearson`, `mean_error`, `ieee1708_grade` |
| `statistic` | `max` / `mean` / `min` for beat-derived metrics; empty otherwise |
| `value` | The number, in the signal's physical unit where one applies |
| `se` | Standard error; `NaN` where none was estimated |
| `n` | Count of underlying units the value was computed over |
| *attribute columns* | `participant`, `recording_id`, `camera_id`, plus whatever store attrs a record carried |

## 5. The records layer

`records.py` normalises the two sources — the trainer's in-memory `windows`
list and one-or-many `*_outputs.pickle` files from a sweep — into one
`WindowRecord` sequence plus run metadata (`fs`, `traces`, `label_norms`).

It is the **single** place the physical-unit inversion happens, using
`label_transforms.INVERSES` and the `label_stats` each record already
carries. Nothing downstream of `records.py` ever sees normalised space. The
three duplicated `_to_physical` implementations collapse into this one.

## 6. Per-sample metadata gains an `attrs` map

Per-sample metadata is currently exactly
`{recording_id, camera_id, start_frame}` (`neural_methods/batch.py`).
Participant, posture, light and session are not carried — the prototypes
recovered them by string-splitting `P015_S01_R3_0_D`, which is
Neckflix-specific and brittle, and `tools/summarise_neckflix_outputs.py`
still derives `participant` that way today.

**Change**: add one generic field, `attrs: dict[str, str]`, filled by
`BaseZarrDataset` from the store's own root attrs — not a fixed list of
Neckflix keys, whatever that store carries. This mirrors exactly what
`DATA.FILTERS` already does, yields the correctly-normalised `participant`
(unprefixed, per the cache contract) for free, and means a second dataset's
grouping needs no report changes.

This edits the shared batch contract, so it is sequenced **first** in the
implementation plan and lands on its own.

## 7. The cardiac clock and the beat layer

**Clock.** Derived once per section from the best available reference in that
recording: ECG R-peaks where the recording carries ECG, ABP feet otherwise.
Deriving the clock separately from the signal being scored means a recording
with CVP+ECG and no ABP still gets a beat segmentation.

One wrinkle when the clock comes from ECG: ECG is shape-class and z-scored
**per window**, so its stitched trace has a step at every window seam and
detecting across the seams would invent R-peaks that are not there. Peak
*locations*, however, are unaffected by a per-window affine rescale. So an
ECG clock is detected per window and the resulting indices are offset into the
section timeline, rather than detected on a stitched trace. An ABP clock,
being `raw`, is detected on the stitched trace directly.

**Beat boundaries from a pressure trace.** Reuse the `find_peaks` the
PhysHydra-era analysis used — the mature version in
`evaluation/prototypes/neckflix_metrics.ipynb`, the one carrying `clip_ends`,
not the earlier copy in `metrics.ipynb`. It is non-maximum suppression over a
sliding window: `max_pool1d_with_indices` of width `width`, keeping the points
that are the extremum of their own neighbourhood. At its established setting
`width = int(fs × 2/3)` (≈ 0.67 s) that enforces a minimum beat separation
directly — the same job `scipy.signal.find_peaks(distance=...)` does — and
`clip_ends` drops a first or last peak lying more than 2 SD off the interior
mean, which is the padding-edge artifact handled. It is proven on this data;
it is lifted as-is into `beats.py`, with `type='min'` giving the feet.

Beats run foot to foot, the standard arterial convention: feet from
`find_peaks(type='min')` on the reference are the beat boundaries, and
`max` / `mean` / `min` of both traces are read inside each
`[foot_i, foot_{i+1})`.

**The one gap, and where it is closed.** The function has no amplitude or
prominence gate, so on a flat or noise-only trace it returns one "peak" per
`width` window rather than none. That is harmless in the reference-anchored
path — an arterial line always has beats — but it is exactly wrong for the
detection-quality pass below, where "did the model produce recognisable beats
at all" is the question being asked. So `beats.py` adds a small amplitude gate
(peak-to-trough excursion against the trace's own noise floor) applied **only**
in the detection-quality pass. The reference-anchored path uses the function
unmodified.

**Per-beat statistics.** Uniformly `max`, `mean`, `min` of both traces over
`[foot_i, foot_{i+1})`, applied to the **reference-defined** intervals for
prediction and reference alike (D5). Display labels come from the signal
registry:

| Signal | `max` | `mean` | `min` |
| --- | --- | --- | --- |
| ABP | systolic | MAP | diastolic |
| CVP | peak | mean | trough |
| SPO2 | max | mean | min |

so the machinery is one code path and no report prints "systolic CVP".
`neural_methods/signals.py` gains a `beat_labels` entry per absolute signal,
beside the existing `class` / `unit` / `prior` / `scale`.

**Stitching, and why the section level falls out of it.** A 5 s window at
about 1 beat/s holds roughly five beats; detecting per window would discard
the partial beats at both edges — a systematic third of them. But the
reference is genuinely continuous: ABP and CVP load `raw`, so contiguous
windows reconstruct the true trace exactly. So: group records by
`(recording_id, camera_id)`, sort by `start_frame`, split into **maximal
contiguous runs**, and detect beats on each run's stitched reference.

Each such run *is* the section level — it falls out of `RANDOM_WINDOWS` and
`STRIDE_SECONDS` rather than being parsed from a filename as the prototypes
did. Beats straddling a window seam are flagged, so the seam effect is
measurable rather than hidden.

**Boundary**: the beat layer applies **only to absolute-class signals loaded
`raw`**. A z-scored shape signal has per-window stats, so stitching it would
produce steps, and it wants waveform and rate metrics rather than pressure
levels.

**Detection quality** is a separate pass: independent detection on the
prediction, matched to reference beats within a tolerance, reported at section
level as sensitivity, PPV and inter-beat-interval error. Never folded into the
agreement statistics, so a model producing no recognisable beats is visibly
bad rather than quietly flattered.

## 8. Levels

| Level | Unit | Populated by |
| --- | --- | --- |
| `beat` | one reference beat | clinical family (max / mean / min error) |
| `window` | one model output window | waveform + rate families |
| `section` | maximal contiguous run of windows | aggregation; seam and detection diagnostics |
| `recording` | `(recording_id, camera_id)` | aggregation — each camera view is a separate device measurement |
| `participant` | the `participant` attr | ISO 81060-3 per-subject mean and SD; IEEE 1708 per-subject MAE |
| `cohort` | everything pooled | pooled bands, grading |

`levels.py` is small and declarative: each level names its grouping keys.
A run populates every level it can; a LOSO fold simply has one participant.

## 9. Metric families

`metrics/__init__.py` holds the registry mapping signal class to applicable
families. Nothing in a YAML selects them (D7).

- **`waveform.py`** — MAE, RMSE, Pearson, CCC, MACC, at window level and
  aggregated upward. In physical units where the signal has them.
- **`rate.py`** — HR by FFT or peak detection, and SNR. This is where
  `report_hr_metrics` lands, keeping `post_process.calculate_metric_per_video`
  as its engine. Applies to every signal (an HR is derivable from ABP as well
  as from ECG), which is what the current per-signal report already does.
- **`clinical.py`** — per-beat agreement for absolute signals, and the generic
  evaluator that applies the criteria table from `standards.py`.

## 10. Uncertainty

`uncertainty.py` carries the prototypes' two estimators, consumed rather than
reinvented: a HAC (Newey-West) standard error with Bartlett / Parzen /
quadratic-spectral kernels, and a moving-block bootstrap.

Take them from `neckflix_metrics.ipynb`, not `metrics.ipynb` — the former is
the mature version and the difference matters. Its `mean_se` returns
`(mean, sd, se)` rather than `(mean, se)`, carries a `naive` / `HAC` switch so
a caller can be explicit about which assumption it is making, and implements
**Andrews (1991) automatic bandwidth selection** per kernel instead of the
earlier fixed `n_sec = 5` lag. Its `get_snr`, `get_macc` and `get_hr_fft` are
also windowed with overlap and return a spread as well as a point estimate,
and `get_snr` additionally reports `auto_snr` — the reference's SNR evaluated
at the *predicted* rate, which the PhysHydra analysis used to rank windows.
All of that comes across.

Applied per D6:

- **Clinical numbers** use the standards' own prescribed aggregation — per
  subject, then pooled — with nothing layered on top.
- **DL-side metrics** computed over autocorrelated samples within a recording
  carry a HAC standard error. A naive `std/sqrt(n)` over 150 autocorrelated
  samples is optimistic by a large factor.
- **Pearson and CCC**, which have no usable closed form here, use the block
  bootstrap. This is the one genuinely expensive computation in the report and
  is therefore the thing `TEST.REPORT.BOOTSTRAP` gates.

## 11. Standards: thresholds, provenance, and what we cannot satisfy

`metrics/standards.py` holds every threshold, grade band and clause reference
as a named constant with a source comment, plus a small table binding each
criterion to the level it consumes and the statistic it tests. Applying them
is one generic evaluator, so ESH or ISO 81060-2 later is rows, not code.

**Provenance is printed.** Every report carries a line naming the standard
revision each threshold came from. No number is quoted without saying where it
came from.

**The thresholds in this design are written from general knowledge of these
standards, not from their text.** Verifying them against the purchased
documents is a named, blocking step in the implementation plan — see §17.

**Computable from our data**: per-subject mean error and SD for the max / mean
/ min triple; pooled bands across subjects; IEEE 1708 MAE grading per subject
and pooled; beat coverage; detection quality.

**Not satisfiable by construction**, reported as explicit notes carrying the
numbers we do have rather than silently omitted: subject count (the standards
require far more than a research cohort), pressure-range distribution across
the cohort, reference-device protocol, cuff procedure, arm-circumference
spread. A research dataset is not a validation study, and the report says so
in its own voice rather than letting a reader infer a clinical claim from a
printed grade.

## 12. Config surface

`TEST.METRICS` is deleted, along with `DEFAULT_METRICS` and
`SUPPORTED_METRICS`. What applies to a signal follows from its class.
`TEST.REPORT` replaces it, gating only what costs something:

```yaml
TEST:
  REPORT:
    BOOTSTRAP: 0                              # resamples for Pearson/CCC standard errors; 0 = skip
    PLOTS: [waveforms, agreement, clinical]   # omit for the standard set
```

The config churn is near zero, which is itself evidence the key was never
pulling its weight: no config under `configs/neckflix/` sets `TEST.METRICS` at
all. The only occurrence in the tree is a comment in
`NECKFLIX_PHYSMAMBA.yaml` recording that the key was omitted. That comment is
deleted and nothing else in `configs/` changes.

## 13. Outputs

| Artifact | Location | Content |
| --- | --- | --- |
| `<id>_metrics.csv` | `config.RUN.output_dir` | the tidy frame, every level |
| `<id>_report.json` | `config.RUN.output_dir` | digest, threshold provenance, unmet study-design notes |
| figures | `LOG_PATH/<exp>/plots` | as today, plus the clinical set |

The printed digest stays a fixed, readable summary; the CSV is where you go
for everything else.

## 14. Failure behaviour

Every one of these is a reported number, never a crash:

- a signal no window carried — the existing "no windows carried this label"
  line is kept;
- a window below the `filtfilt` pad length — the rate family skips it, other
  families continue;
- a section where no beats are detectable — counted as beat coverage;
- a degenerate correlation — the current wording is preserved verbatim
  ("undefined, ground-truth HR is constant across all n windows"), because it
  is better than `nan +/- nan`;
- non-contiguous records — sections simply come out shorter;
- **a participant level with n = 1**, which is every LOSO fold — criteria
  needing a cohort print "not computable at n = 1" rather than a grade that
  would be meaningless.

## 15. Tests

Four, replacing `tests/test_metrics_report.py` rather than sitting beside it,
consistent with the repo's minimal-testing rule:

1. **Tidy-frame contract** — columns, dtypes, and the level vocabulary.
2. **Beat layer** — a synthetic arterial waveform with known systolic,
   diastolic and rate; beats found at the right count, values recovered within
   tolerance.
3. **Criteria** — a hand-built error distribution with known mean and SD must
   produce the expected IEEE 1708 grade and 81060-3 verdict.
4. **Round trip** — the trainer's in-memory records and the same run's pickle
   produce identical frames.

Plus the existing smoke run through `NECKFLIX_*_SMOKE.yaml`.

## 16. Deletions, caller migration, and one observation

**Deleted**: `evaluation/metrics_report.py` and `evaluation/BlandAltmanPy.py`
(content redistributed into `report.py` and `plots.py`; `BlandAltmanPy` takes
a `config` object purely to locate an output directory and prints errors
instead of raising, so none of its behaviour is worth carrying forward);
`evaluation/prototypes/` once consumed, closing Phase 7 step 3;
`tests/test_metrics_report.py`.

**Callers that must move in the same change**:
`neural_methods/trainer/MultiSignalTrainer.py` and
`unsupervised_methods/unsupervised_predictor.py` — the unsupervised path
scores per trace through `report_hr_metrics` today and must land on the rate
family. `tools/summarise_neckflix_outputs.py` becomes a thin CLI over
`evaluation.report` that also accepts a sweep directory (D3).

**Observation, not scope**: `PerSignalLoss` gives every absolute-class signal
the same soft `MAX` / `MIN` peak terms, so CVP is trained against a
systolic/diastolic-shaped objective for the same reason the old plot labelled
it that way. That is the training objective rather than evaluation, and
changing it is a separate decision with its own retro. Recorded here so the
next person to read both modules sees it.

## 17. Verification steps carried into the implementation plan

1. **Confirm every threshold, grade band and aggregation formula in
   `metrics/standards.py` against the purchased ISO 81060-3:2022 and
   IEEE 1708-2014 / 1708a-2019 texts** before any report prints a pass/fail.
   Blocking for clinical claims; not blocking for the rest of the build.
2. Confirm the beat detector's constraints (refractory window, prominence
   scaling) against real cached ABP rather than only the synthetic test
   waveform.
3. Confirm that the `attrs` metadata addition leaves the batch contract tests
   green before anything else in the plan proceeds.
