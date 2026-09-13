"""Heart rate from every cardiac trace, and from all of them fused.

Per reading, for every trace the registry marks cardiac (PPG, ECG, ABP, CVP)
that the reading labels, the estimate the upstream toolbox made: detrend,
bandpass to the heart-rate band, periodogram, the largest in-band bin, in
beats per minute. Run on the label and on the prediction, so every trace
reports its own reference and predicted rate and is compared with itself.
Two more *sources* join the per-trace ones whenever a reading carries at
least two cardiac traces:

``FUSED``
    The traces' power spectra, each normalised to unit power in the band,
    combined as their geometric mean — a product of spectra, so the
    frequency the traces agree on wins and a peak only one of them has
    (CVP's respiratory harmonic, say) is suppressed. Power spectra rather
    than complex ones: the traces are phase-shifted against each other by
    transit time and morphology, and a complex sum would cancel at the very
    fundamental it is after. Nothing goes back to a waveform, because a rate
    needs no phase. The prediction side fuses exactly the traces the label
    side has, so the fused prediction is compared with a fused label built
    the same way.

``MEDIAN``
    The median of the per-trace rates: the cheap baseline the fusion has to
    beat.

No config. The band and the detrender are the upstream ones; the label
modes are ``raw`` / ``zscore`` / ``minmax`` and the trace tables are already in
physical units, so nothing here needs to know how a trace was preprocessed.
The three upstream helpers this needs (the smoothness-prior detrender, the
FFT length and the maximum amplitude of cross-correlation) live at the top
of this module; the rest of the toolbox's post-processing is gone.
"""

import functools

import numpy as np
from scipy.linalg import solveh_banded
from scipy.signal import butter, filtfilt, periodogram
from scipy.sparse import diags as sparse_diags

#: The heart-rate band in Hz, upstream's 36–198 bpm.
BAND = (0.6, 3.3)
DETREND_LAMBDA = 100
#: Half-width of the harmonic bins counted as signal in the SNR, in bpm.
SNR_DEVIATION_BPM = 6
#: ``filtfilt`` pads ``3 * max(len(a), len(b))`` = 9 samples for the
#: first-order bandpass and needs strictly more than that to work on.
MIN_FRAMES = 10
#: Floor under a normalised spectrum before its log, so one trace's empty bin
#: cannot veto the frequency every other trace favours.
SPECTRUM_FLOOR = 1e-12

FUSED, MEDIAN = "FUSED", "MEDIAN"
RATE_METRICS = ("ref_hr", "pred_hr", "err_hr", "snr", "macc")
_NAN = float("nan")


# ---------------------------------------------------------------------------
# The upstream toolbox's helpers
# ---------------------------------------------------------------------------
def next_power_of_2(x: int) -> int:
    """The smallest power of two at or above ``x``."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@functools.lru_cache(maxsize=32)
def _detrend_bands(signal_length: int, lambda_value: float) -> np.ndarray:
    """Upper-banded form of ``I + lambda^2 * D'D`` for the smoothness prior.

    ``D`` is the second-difference operator, so ``D'D`` is symmetric and
    pentadiagonal: only three diagonals are needed, and they depend on nothing
    but the length, hence the cache.
    """
    second_difference = sparse_diags(
        [1.0, -2.0, 1.0], [0, 1, 2], shape=(signal_length - 2, signal_length))
    gram = (second_difference.T @ second_difference).tocsr()
    banded = np.zeros((3, signal_length))
    banded[2] = 1.0 + lambda_value ** 2 * gram.diagonal(0)
    banded[1, 1:] = lambda_value ** 2 * gram.diagonal(1)
    banded[0, 2:] = lambda_value ** 2 * gram.diagonal(2)
    return banded


def detrend(trace, lambda_value: float) -> np.ndarray:
    """Tarvainen's smoothness-prior detrender: removes the smooth trend ``z``
    that solves ``(I + lambda^2 D'D) z = x`` and returns ``x - z``, solved as
    the banded system it is (O(n), not a dense inverse). A trace too short
    to difference is returned as is."""
    trace = np.asarray(trace, dtype=np.float64)
    length = trace.shape[0]
    if length < 3:
        return trace
    flat = trace.reshape(length, -1)
    trend = solveh_banded(_detrend_bands(length, float(lambda_value)), flat, lower=False)
    return (flat - trend).reshape(trace.shape)


def macc(pred, ref) -> float:
    """Maximum amplitude of cross-correlation: the largest
    ``|corrcoef(pred, roll(ref, lag))|`` over every lag, computed as one
    circular cross-correlation via FFT (the two norms are roll-invariant, so
    this equals the per-lag loop). Zero for a constant trace or fewer than
    two samples."""
    pred = np.squeeze(np.asarray(pred, dtype=np.float64))
    ref = np.squeeze(np.asarray(ref, dtype=np.float64))
    n = min(pred.size, ref.size)
    pred, ref = pred[:n], ref[:n]
    if n < 2:
        return 0.0
    centred_pred, centred_ref = pred - pred.mean(), ref - ref.mean()
    denominator = np.sqrt((centred_pred ** 2).sum() * (centred_ref ** 2).sum())
    if denominator == 0:
        return 0.0
    # circular[lag] == sum_i centred_pred[i] * centred_ref[(i - lag) % n]
    circular = np.fft.irfft(
        np.fft.rfft(centred_pred) * np.conj(np.fft.rfft(centred_ref)), n=n)
    # Lags 0 .. n-2, matching upstream's range(0, len(pred) - 1).
    return float(np.max(np.abs(circular[:n - 1])) / denominator)


# ---------------------------------------------------------------------------
# One trace to one spectrum to one rate
# ---------------------------------------------------------------------------
def clean(trace, fs: float) -> np.ndarray:
    """Detrended and zero-phase bandpassed to the heart-rate band."""
    detrended = detrend(np.asarray(trace, dtype=np.float64), DETREND_LAMBDA)
    b, a = butter(1, [BAND[0] / fs * 2, BAND[1] / fs * 2], btype="bandpass")
    return filtfilt(b, a, detrended)


def spectrum(trace, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, power)`` of a cleaned trace, zero-padded to a power of two."""
    trace = np.asarray(trace, dtype=np.float64)
    return periodogram(trace, fs=fs, nfft=next_power_of_2(trace.size), detrend=False)


def in_band(freqs) -> np.ndarray:
    return (freqs >= BAND[0]) & (freqs <= BAND[1])


def rate_of(freqs, power) -> float:
    """The largest in-band bin, in beats per minute."""
    band = in_band(freqs)
    return float(freqs[band][np.argmax(power[band])] * 60)


def normalised(freqs, power) -> np.ndarray:
    """The spectrum scaled to unit power inside the band, so every trace
    weighs the same regardless of its units and however much power sits at
    DC or in the respiratory range."""
    total = power[in_band(freqs)].sum()
    return power / total if total > 0 else power


def fuse(freqs, powers) -> np.ndarray:
    """Geometric mean of the traces' normalised spectra, bin by bin."""
    stacked = np.stack([normalised(freqs, p) for p in powers])
    return np.exp(np.log(np.maximum(stacked, SPECTRUM_FLOOR)).mean(axis=0))


def snr(freqs, power, hr_bpm: float) -> float:
    """Power within +/- 6 bpm of the reference rate and its second harmonic,
    over the rest of the band, in dB (upstream's definition)."""
    deviation = SNR_DEVIATION_BPM / 60
    harmonic = np.zeros_like(freqs, dtype=bool)
    for centre in (hr_bpm / 60, 2 * hr_bpm / 60):
        harmonic |= (freqs >= centre - deviation) & (freqs <= centre + deviation)
    remainder = in_band(freqs) & ~harmonic
    signal_power, noise_power = power[harmonic].sum(), power[remainder].sum()
    if noise_power <= 0 or signal_power <= 0:
        return _NAN
    return float(10 * np.log10(signal_power / noise_power))


# ---------------------------------------------------------------------------
# One reading to its rows
# ---------------------------------------------------------------------------
def reading_rates(traces: dict, fs: float) -> list:
    """``[{source, ref_hr, pred_hr, err_hr, snr, macc}, ...]`` for one reading,
    given ``{signal: (label, prediction)}`` over the cardiac traces it
    carries, both finite: one row per trace, then ``FUSED`` and ``MEDIAN``
    when there are two or more to combine. Empty for a reading too short
    to filter."""
    rows, ref_powers, pred_powers = [], [], []
    freqs = None
    for sig, (ref, pred) in traces.items():
        ref = np.asarray(ref, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if ref.size < MIN_FRAMES:
            return []
        ref, pred = clean(ref, fs), clean(pred, fs)
        freqs, ref_power = spectrum(ref, fs)
        _, pred_power = spectrum(pred, fs)
        ref_hr, pred_hr = rate_of(freqs, ref_power), rate_of(freqs, pred_power)
        rows.append({"source": sig, "ref_hr": ref_hr, "pred_hr": pred_hr,
                     "err_hr": pred_hr - ref_hr, "snr": snr(freqs, pred_power, ref_hr),
                     "macc": macc(pred, ref)})
        ref_powers.append(ref_power)
        pred_powers.append(pred_power)
    if len(rows) < 2:
        return rows

    ref_fused, pred_fused = fuse(freqs, ref_powers), fuse(freqs, pred_powers)
    ref_hr, pred_hr = rate_of(freqs, ref_fused), rate_of(freqs, pred_fused)
    rows.append({"source": FUSED, "ref_hr": ref_hr, "pred_hr": pred_hr,
                 "err_hr": pred_hr - ref_hr, "snr": snr(freqs, pred_fused, ref_hr),
                 "macc": _NAN})
    ref_hr = float(np.median([row["ref_hr"] for row in rows[:-1]]))
    pred_hr = float(np.median([row["pred_hr"] for row in rows[:-1]]))
    rows.append({"source": MEDIAN, "ref_hr": ref_hr, "pred_hr": pred_hr,
                 "err_hr": pred_hr - ref_hr, "snr": _NAN, "macc": _NAN})
    return rows
