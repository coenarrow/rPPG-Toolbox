"""Heart rate from every cardiac trace, and from all of them fused.

Per window, for every trace the registry marks cardiac (PPG, ECG, ABP, CVP)
that the window labels, the estimate the upstream toolbox made: detrend,
bandpass to the heart-rate band, periodogram, the largest in-band bin, in
beats per minute. Run on the label and on the prediction, so every trace
reports its own reference and predicted rate and is compared with itself.
Two more *sources* join the per-trace ones whenever a window carries at
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
modes are ``raw`` / ``zscore`` / ``minmax`` and the records are already in
physical units, so nothing here needs to know how a trace was preprocessed.
"""

import numpy as np
from scipy.signal import butter, filtfilt, periodogram

from src.evaluation.post_process import _compute_macc, _detrend, _next_power_of_2
from neural_methods.signals import is_cardiac

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
# One trace to one spectrum to one rate
# ---------------------------------------------------------------------------
def clean(trace, fs: float) -> np.ndarray:
    """Detrended and zero-phase bandpassed to the heart-rate band."""
    detrended = _detrend(np.asarray(trace, dtype=np.float64), DETREND_LAMBDA)
    b, a = butter(1, [BAND[0] / fs * 2, BAND[1] / fs * 2], btype="bandpass")
    return filtfilt(b, a, detrended)


def spectrum(trace, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """``(frequencies, power)`` of a cleaned trace, zero-padded to a power of two."""
    trace = np.asarray(trace, dtype=np.float64)
    return periodogram(trace, fs=fs, nfft=_next_power_of_2(trace.size), detrend=False)


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
# One window to its rows
# ---------------------------------------------------------------------------
def _array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def cardiac_traces(record) -> list:
    """The cardiac traces this window both labels and predicts."""
    return [sig for sig in record["predictions"]
            if is_cardiac(sig) and bool(record["label_mask"][sig])]


def window_rates(record, fs: float) -> list:
    """``[{source, ref_hr, pred_hr, err_hr, snr, macc}, ...]`` for one window:
    one per cardiac trace, then ``FUSED`` and ``MEDIAN`` when there are two
    or more to combine. Empty for a window too short to filter."""
    traces = cardiac_traces(record)
    if not traces:
        return []
    rows, ref_powers, pred_powers = [], [], []
    freqs = None
    for sig in traces:
        ref, pred = _array(record["labels"][sig]), _array(record["predictions"][sig])
        if ref.size < MIN_FRAMES:
            return []
        ref, pred = clean(ref, fs), clean(pred, fs)
        freqs, ref_power = spectrum(ref, fs)
        _, pred_power = spectrum(pred, fs)
        ref_hr, pred_hr = rate_of(freqs, ref_power), rate_of(freqs, pred_power)
        rows.append({"source": sig, "ref_hr": ref_hr, "pred_hr": pred_hr,
                     "err_hr": pred_hr - ref_hr, "snr": snr(freqs, pred_power, ref_hr),
                     "macc": _compute_macc(pred, ref)})
        ref_powers.append(ref_power)
        pred_powers.append(pred_power)
    if len(traces) < 2:
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
