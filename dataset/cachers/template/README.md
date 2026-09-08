# Template cacher

The starting point for every dataset cacher in this repository. A cacher
turns one dataset's raw recordings into zarr stores that satisfy
[the cache contract](../../../docs/cache-contract.md) — align, optionally
resize, write — and nothing more: no normalisation, no filtering, no
windowing. Those are the reader's and the model's business.

The structure is the Neckflix cacher's (the submodule at
`dataset/cachers/neckflix/`), with the Neckflix-specific parts hollowed out.
Each cacher is its own `uv` project with its own lock, so decode
dependencies (`av`, `opencv`, `h5py`, ...) never enter the training
environment. Run one with `uv run --project dataset/cachers/<name> ...`.

## Making a cacher from this template

1. Copy `dataset/cachers/template/` to `dataset/cachers/<dataset>/`.
2. Rename: `template_cacher` → `<dataset>_cacher` (the `src/` package, its
   imports, `pyproject.toml`'s `packages` entry) and `template-preprocess`
   → `<dataset>-preprocess` (the script name, `prog=` in `cli.py`). Add the
   decode dependencies you need to `pyproject.toml`, then `uv lock`.
3. Fill in the marked places. Each docstring states the contract it must
   meet; nothing else needs touching.

| Where | What |
| --- | --- |
| `scan.py` `discover_recordings` | list the dataset's recording names |
| `scan.py` `scan_recording` | metadata-only probe: root attrs (**`participant` as a string, verbatim**), perspectives → modalities → streams, source resolution |
| `traces.py` `TRACE_UNITS` | the traces the dataset carries and their units |
| `traces.py` `read_stream_traces` | per-frame timestamps (µs) and frame-rate traces for one stream |
| `video.py` `decode_video` | frames as `(C, T, H, W)`, resized per frame |
| `writer.py` `NOMINAL_FPS` | the perspective's nominal frame rate |
| `cli.py` `MODALITIES`, `PERSPECTIVES` | what the dataset can write |

4. Write the dataset's cache spec, `dataset/data_loader/<DATASET>.md`: how
   the raw files map onto the layout, which root attrs it writes and how
   their values are spelled (downstream configs filter on them exactly as
   written), and any liberty taken (synthesised timestamps, resampled
   traces).
5. Acceptance test: build a cache and run the validator over it from the
   training environment. All PASS, or the cacher is not done.

```bash
uv run --project dataset/cachers/<dataset> <dataset>-preprocess \
    --input-dir <raw root> --output-dir <cache dir> --resize 128 128
uv run python tools/validate_cache.py <cache dir>
```

## What stays as it is

`writer.py` (the schema, compression and chunking, identical to Neckflix so
every cache reads alike, and the write-time refusal of a non-string
`participant`), the alignment arithmetic in `traces.py`, and all of `cli.py`
below its two constants: the skip/rebuild bookkeeping (`complete` and
`resized_to` root attrs — the cacher's own, not the contract's), the
parallel runner, per-recording failure isolation, the summary line and exit
status.

## What the template deliberately leaves out

- Anything not in the contract vocabulary. An extra perspective type (the
  Neckflix event camera) is a contract change first, a cacher change second.
- Docker packaging and a CI image build. Add them when a cacher needs a
  codec the training environment cannot build, as Neckflix does for ECF.
- Tests in the training suite. The validator is the test.
