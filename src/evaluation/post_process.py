"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, power2db etc.
"""

import functools

import numpy as np
import scipy
import scipy.io
from scipy.linalg import solveh_banded
from scipy.signal import butter
from scipy.sparse import diags as sparse_diags


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@functools.lru_cache(maxsize=32)
def _detrend_bands(signal_length, lambda_value):
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


def _detrend(input_signal, lambda_value):
    """Detrend PPG signal (Tarvainen smoothness priors).

    Removes the smooth trend ``z`` that solves
    ``(I + lambda^2 D'D) z = x`` and returns ``x - z``. Mathematically the same
    as the textbook ``(I - (I + lambda^2 D'D)^-1) x``, but solved as the banded
    system it is: O(n) instead of an O(n^3) dense inverse, which for a
    300-sample window is the difference between ~1 s and well under a
    millisecond — and this runs once per signal per window per method.
    """
    signal_length = input_signal.shape[0]
    if signal_length < 3:
        return np.asarray(input_signal, dtype=np.float64)
    flat = np.asarray(input_signal, dtype=np.float64).reshape(signal_length, -1)
    banded = _detrend_bands(signal_length, float(lambda_value))
    trend = solveh_banded(banded, flat, lower=False)
    return (flat - trend).reshape(np.shape(input_signal))

def power2db(mag):
    """Convert power to db."""
    return 10 * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):
    # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
    # we recommend low_pass=0.75 and high_pass=2.5 instead of the defaults above.
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_ppg_signal(np.array): predicted PPG signal
            label_ppg_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation

    Computed as one circular cross-correlation via FFT rather than a Python
    loop over every lag: ``corrcoef(pred, roll(gt, lag))`` is exactly that
    correlation normalised by the two (roll-invariant) norms, so the result is
    identical while the cost drops from O(n^2) to O(n log n).
    """
    pred = np.squeeze(np.asarray(pred_signal, dtype=np.float64))
    gt = np.squeeze(np.asarray(gt_signal, dtype=np.float64))
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    if min_len < 2:
        return 0.0

    centred_pred = pred - pred.mean()
    centred_gt = gt - gt.mean()
    denominator = np.sqrt((centred_pred ** 2).sum() * (centred_gt ** 2).sum())
    if denominator == 0:
        return 0.0
    # circular_corr[lag] == sum_i centred_pred[i] * centred_gt[(i - lag) % n]
    circular_corr = np.fft.irfft(
        np.fft.rfft(centred_pred) * np.conj(np.fft.rfft(centred_gt)), n=min_len)
    # Lags 0 .. n-2, matching the original np.arange(0, len(pred) - 1).
    return float(np.max(np.abs(circular_corr[:min_len - 1])) / denominator)

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.6 Hz
        to 3.3 Hz. 

        Ref for low_pass and high_pass filters:
        R. Cassani, A. Tiwari and T. H. Falk, "Optimal filter characterization for photoplethysmography-based pulse rate and 
        pulse power spectrum estimation," 2020 IEEE Engineering in Medicine & Biology Society (EMBC), Montreal, QC, Canada,
        doi: 10.1109/EMBC44109.2020.9175396.

        Note: to more closely match results in the NeurIPS 2023 toolbox paper, we recommend low_pass=0.75 and high_pass=2.5 
        instead of the defaults above.

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = power2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.5] Hz, equals [45, 150] beats per min
        # bandpass filter between [0.6, 3.3] Hz, equals [36, 198] beats per min
        #
        # Note: to more closely match results in the NeurIPS 2023 toolbox paper,
        # we recommend using 0.75 in place of 0.6 and 2.5 in place of 3.3 in the 
        # below line.
        [b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))
    
    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr(predictions, fs=fs)
        hr_label = _calculate_fft_hr(labels, fs=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_label, hr_pred, SNR, macc