# PURE cacher

Turns the PURE dataset's raw recordings (timestamped PNG sequences plus a
finger-oximeter JSON sidecar) into zarr stores that satisfy
[the cache contract](../../../docs/cache-contract.md). Built from
[the template](../template/README.md); the mapping it implements is the
dataset's cache spec, [`dataset/data_loader/PURE.md`](../../data_loader/PURE.md).

```bash
uv run --project dataset/cachers/pure pure-preprocess \
    --input-dir D:/PURE/PURE --output-dir <cache dir> --resize 256 256
uv run python tools/validate_cache.py <cache dir>
```

`--input-dir` is the directory holding the `SS-TT` recording directories.
One store per recording, `SS-TT.zarr`, one perspective `1` with one `rgb`
modality and one `ppg` trace. Drop `--resize` to keep the native 640x480.
Reruns skip complete stores; `--overwrite` rebuilds.

What it writes, and why, is in the spec. In one line: frames RGB uint8 in
chronological order; `timestamps_us` the capture times as epoch
microseconds; `ppg` the oximeter waveform linearly interpolated at the
frame times (`units: arb`); root attrs carrying the identity tokens, the
capture time, the measured rates, and a summary of every other oximeter
field the sidecar records.
