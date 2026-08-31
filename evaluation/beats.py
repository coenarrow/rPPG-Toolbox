"""Reference-anchored beats: the clock, the intervals, the per-beat statistics.

The detector is the one the PhysHydra-era analysis used (the mature copy in
``evaluation/prototypes/neckflix_metrics.ipynb``, the one carrying
``clip_ends``): non-maximum suppression over a sliding window, which enforces
a minimum beat separation directly. Two changes on the way in — the reshapes
are einops per the repo rule, and the odd-width normalisation now happens
*before* pooling rather than after, where it had no effect.
"""

from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F

#: What is read off each beat. Uniform across absolute signals — only the
#: display wording is per-signal (``signals.beat_labels``).
BEAT_STATS = ("max", "mean", "min")

#: Sliding-window width as a fraction of the sampling rate: 2/3 s, the setting
#: the PhysHydra analysis ran at. Wide enough to suppress a dicrotic notch,
#: narrow enough to keep every beat up to ~90 bpm.
WIDTH_SECONDS = 2 / 3


def find_peaks(trace, kind="max", width=31, clip_ends=False):
    """Local extrema that survive non-maximum suppression of ``width`` samples.

    Returns ``(indices, values)``. A point is kept when it is the extremum of
    its own neighbourhood, which makes ``width`` a minimum-separation
    constraint rather than a smoothing parameter.
    """
    if kind not in ("max", "min"):
        raise ValueError(f"kind must be 'max' or 'min', got {kind!r}")
    if width % 2 == 0:
        width += 1          # symmetric padding needs an odd kernel
    tensor = torch.as_tensor(np.asarray(trace), dtype=torch.float32)
    work = tensor if kind == "max" else -tensor
    _, indices = F.max_pool1d_with_indices(
        rearrange(work, "t -> 1 1 t"), kernel_size=width, stride=1,
        padding=width // 2)
    indices = rearrange(indices, "1 1 t -> t")
    candidates = indices.unique()
    kept = candidates[indices[candidates] == candidates]
    idx = np.atleast_1d(kept.numpy().astype(int))
    values = np.atleast_1d(tensor.numpy()[idx].astype(np.float64))
    if clip_ends and len(values) > 3:
        interior_mean = float(values[1:-1].mean())
        interior_std = float(values[1:-1].std())
        if abs(values[0] - interior_mean) > 2 * interior_std:
            idx, values = idx[1:], values[1:]
        if abs(values[-1] - interior_mean) > 2 * interior_std:
            idx, values = idx[:-1], values[:-1]
    return idx, values


def beat_intervals(reference, fs) -> list:
    """Foot-to-foot beat boundaries, detected on the reference trace only.

    Anchoring on the reference means every reference beat yields exactly one
    comparison, so agreement statistics carry no selection bias from a
    prediction whose beats are hard to find.
    """
    width = max(3, int(fs * WIDTH_SECONDS))
    feet, _ = find_peaks(reference, kind="min", width=width, clip_ends=True)
    return [(int(start), int(end)) for start, end in zip(feet[:-1], feet[1:])
            if end - start >= 3]


def beat_stats(trace, intervals) -> dict:
    """``max`` / ``mean`` / ``min`` of ``trace`` inside each beat interval."""
    values = np.asarray(trace, dtype=np.float64)
    beats = [values[start:end] for start, end in intervals]
    if not beats:
        return {name: np.array([], dtype=np.float64) for name in BEAT_STATS}
    return {
        "max": np.array([beat.max() for beat in beats]),
        "mean": np.array([beat.mean() for beat in beats]),
        "min": np.array([beat.min() for beat in beats]),
    }


#: A prediction must repeat at a plausible heart rate before its extrema count
#: as beats. Non-maximum suppression alone always returns one extremum per
#: window, so without this a model emitting noise would score a full
#: complement of beats. Autocorrelation is the discriminator rather than raw
#: amplitude: amplitude ratios separate noise from signal by only a few per
#: cent here, periodicity separates them by an order of magnitude.
PULSATILITY_THRESHOLD = 0.3

#: Lag window the pulsatility test searches, in seconds — 30 to 240 bpm.
PULSATILITY_LAGS = (0.25, 2.0)


def _has_pulsatility(trace, fs) -> bool:
    """True when the trace repeats at a plausible heart rate.

    The normalised autocorrelation of white noise over a few hundred samples
    peaks around 0.1; a pulsatile trace peaks near 1.0 at its beat lag.
    """
    values = np.asarray(trace, dtype=np.float64)
    if values.size < 8:
        return False
    centred = values - values.mean()
    energy = float(centred @ centred)
    if energy <= 0:
        return False
    # FFT autocorrelation: a section can be thousands of samples, and the
    # direct form is quadratic in that length.
    size = 1 << int(np.ceil(np.log2(2 * centred.size)))
    spectrum = np.fft.rfft(centred, n=size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=size)[:centred.size]
    correlation /= energy
    low, high = PULSATILITY_LAGS
    first, last = int(low * fs), min(int(high * fs), correlation.size - 1)
    if first > last:
        return False
    return bool(correlation[first:last + 1].max() >= PULSATILITY_THRESHOLD)


def detection_quality(prediction, reference, fs, tolerance_seconds=0.15) -> dict:
    """How well the prediction's *own* beats line up with the reference's.

    Kept apart from the agreement statistics on purpose. Agreement is
    reference-anchored, so it says nothing about whether the prediction has
    recognisable beats; this does, and folding the two together would let a
    model that finds few beats look accurate on the ones it did find.
    """
    width = max(3, int(fs * WIDTH_SECONDS))
    reference_feet, _ = find_peaks(reference, kind="min", width=width,
                                   clip_ends=True)
    if _has_pulsatility(prediction, fs):
        predicted_feet, _ = find_peaks(prediction, kind="min", width=width,
                                       clip_ends=True)
    else:
        predicted_feet = np.array([], dtype=int)

    tolerance = tolerance_seconds * fs
    unmatched = list(map(int, reference_feet))
    matched = 0
    for foot in predicted_feet:
        if not unmatched:
            break
        nearest = min(unmatched, key=lambda candidate: abs(candidate - foot))
        if abs(nearest - foot) <= tolerance:
            unmatched.remove(nearest)
            matched += 1

    n_reference, n_detected = len(reference_feet), len(predicted_feet)
    reference_ibi = np.diff(reference_feet) / fs
    predicted_ibi = np.diff(predicted_feet) / fs
    ibi_error = (abs(float(predicted_ibi.mean()) - float(reference_ibi.mean()))
                 if predicted_ibi.size and reference_ibi.size else float("nan"))
    return {
        "sensitivity": matched / n_reference if n_reference else float("nan"),
        "ppv": matched / n_detected if n_detected else 0.0,
        "ibi_error": ibi_error,
        "n_reference": n_reference,
        "n_detected": n_detected,
    }
