"""The seven traditional rPPG methods: numerics, NumPy-2 safety, dict contract.

POS, ICA and PBV were rewritten off APIs NumPy 2.0 removed or changed
(``np.mat``; ``linalg.solve``'s vector-stack broadcasting), and POS was
vectorised. The first group of tests pins them to the frozen pre-NumPy-2
originals in ``tests/reference/`` so the rewrites are equivalences, not
reimplementations.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from evaluation.post_process import _calculate_fft_hr, calculate_metric_per_video
from tests.reference.pos_ica_numpy1 import ICA_POH_REF, POS_WANG_REF
from unsupervised_methods import utils
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.ICA_POH import ICA_POH
from unsupervised_methods.methods.LGI import LGI
from unsupervised_methods.methods.OMIT import OMIT
from unsupervised_methods.methods.PBV import PBV
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.unsupervised_predictor import (
    BVP_ESTIMATORS, MIN_WINDOW, _dict_windows, estimate_bvp,
)

FS = 30
ALL_METHODS = ("POS", "CHROM", "ICA", "GREEN", "LGI", "PBV", "OMIT")


def pulsatile_clip(n_frames=300, hr_bpm=72, seed=0, hw=(8, 6)):
    """A synthetic clip whose green channel carries a clean pulse at ``hr_bpm``."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / FS
    pulse = np.sin(2 * np.pi * (hr_bpm / 60) * t)
    amplitudes = np.array([2.0, 6.0, 1.5])
    frames = 120 + pulse[:, None, None, None] * amplitudes
    return frames + rng.normal(0, 0.3, (n_frames, *hw, 3))


# --- equivalence with the frozen pre-NumPy-2 originals -------------------
@pytest.mark.parametrize("n_frames", [120, 180, 257, 300])
def test_pos_matches_frozen_original(n_frames):
    rng = np.random.default_rng(n_frames)
    frames = rng.uniform(20, 230, (n_frames, 9, 7, 3))
    assert np.allclose(POS_WANG(frames, FS), POS_WANG_REF(frames, FS), atol=1e-10)


@pytest.mark.parametrize("n_frames", [120, 180, 257])
def test_ica_matches_frozen_original(n_frames):
    rng = np.random.default_rng(n_frames)
    frames = rng.uniform(20, 230, (n_frames, 9, 7, 3))
    assert np.allclose(ICA_POH(frames, FS), ICA_POH_REF(frames, FS), atol=1e-8)


def test_pbv_matches_pre_numpy2_solve_semantics():
    """NumPy < 2 treated a 2-D RHS as a stack of vectors; spell that out."""
    rng = np.random.default_rng(3)
    frames = rng.uniform(20, 230, (200, 8, 6, 3))
    rgb = utils.process_video(frames)
    norm = rgb / np.expand_dims(rgb.mean(axis=2), 2)
    pbv_n = np.array([norm[:, c, :].std(axis=1) for c in range(3)])
    pbv_d = np.sqrt(sum(norm[:, c, :].var(axis=1) for c in range(3)))
    pbv = pbv_n / pbv_d                                          # (3, 1)
    channels_last = np.swapaxes(norm, 1, 2)
    Q = norm @ channels_last
    W = np.linalg.solve(Q, np.swapaxes(pbv, 0, 1)[..., None])[..., 0]
    numerator = channels_last @ np.expand_dims(W, 2)
    denominator = np.swapaxes(np.expand_dims(pbv.T, 2), 1, 2) @ np.expand_dims(W, 2)
    expected = (numerator / denominator).squeeze(axis=2).reshape(-1)
    assert np.allclose(PBV(frames), expected, atol=1e-10)


def test_no_method_touches_removed_numpy_aliases():
    assert not hasattr(np, "mat"), "test premise: np.mat is gone in NumPy 2"
    frames = pulsatile_clip(n_frames=180)
    for method in ALL_METHODS:
        estimate_bvp(method, frames, FS)      # would raise AttributeError pre-fix


# --- behaviour ----------------------------------------------------------
@pytest.mark.parametrize("method", ALL_METHODS)
def test_method_returns_finite_signal(method):
    frames = pulsatile_clip()
    bvp = estimate_bvp(method, frames, FS)
    assert bvp.ndim == 1
    assert np.isfinite(bvp).all()
    # CHROM drops a partial trailing window; nothing may be longer than the clip.
    assert MIN_WINDOW <= len(bvp) <= frames.shape[0]


@pytest.mark.parametrize("method", ALL_METHODS)
def test_method_recovers_a_synthetic_heart_rate(method):
    """Through the real evaluation path (detrend + bandpass + FFT)."""
    frames = pulsatile_clip(n_frames=600, hr_bpm=72)
    reference = np.sin(2 * np.pi * (72 / 60) * np.arange(600) / FS)
    bvp = estimate_bvp(method, frames, FS)
    gt_hr, pred_hr, _snr, macc = calculate_metric_per_video(
        bvp, reference[:len(bvp)], diff_flag=False, fs=FS, hr_method='FFT')
    assert abs(gt_hr - 72) < 3
    assert abs(pred_hr - 72) < 5, f"{method} recovered {pred_hr:.1f} bpm"
    assert macc > 0.5


@pytest.mark.parametrize("method", ALL_METHODS)
def test_precomputed_rgb_trace_is_equivalent_to_the_clip(method):
    """The predictor reduces each clip once and hands every method the trace."""
    frames = pulsatile_clip()
    from_clip = estimate_bvp(method, frames, FS)
    from_trace = estimate_bvp(method, utils.rgb_trace(frames), FS)
    assert np.allclose(from_clip, from_trace, atol=1e-9)


def test_rgb_trace_rejects_odd_shapes():
    with pytest.raises(ValueError, match="Expected frames"):
        utils.rgb_trace(np.zeros((4, 5, 3)))


def test_estimate_bvp_names_the_known_methods():
    with pytest.raises(ValueError, match="unsupervised method name wrong"):
        estimate_bvp("MAGIC", pulsatile_clip(60), FS)
    assert set(BVP_ESTIMATORS) == set(ALL_METHODS)


# --- dict contract ------------------------------------------------------
def _neckflix_sample(*, signals=("ABP", "CVP"), present=("ABP",),
                     channels=("R", "G", "B"), rgb_present=True, t=64):
    clip = pulsatile_clip(n_frames=t, hw=(5, 4))
    return {
        "frames": {c: torch.from_numpy(clip[..., i]).float().unsqueeze(0)
                   for i, c in enumerate(channels)},
        "labels": {s: torch.randn(t) for s in signals},
        "label_stats": {s: {k: torch.tensor(0.0) for k in ("mean", "std", "min", "max")}
                        for s in signals},
        "channel_mask": {c: torch.tensor(rgb_present) for c in channels},
        "label_mask": {s: torch.tensor(s in present) for s in signals},
        "metadata": {"recording_id": "P001_S01_R1_0_D", "camera_id": "1",
                     "start_frame": 0},
    }


def test_dict_windows_yields_a_trace_and_only_present_labels():
    batch = default_collate([_neckflix_sample()])
    (trace, references, name), = list(_dict_windows(batch))
    assert trace.shape == (64, 3)
    assert set(references) == {"ABP"}          # CVP is masked out
    assert name.startswith("P001_S01_R1_0_D_cam1@")


def test_dict_windows_skips_windows_with_no_present_label():
    batch = default_collate([_neckflix_sample(present=())])
    assert list(_dict_windows(batch)) == []


def test_dict_windows_skips_zero_filled_rgb():
    batch = default_collate([_neckflix_sample(rgb_present=False)])
    assert list(_dict_windows(batch)) == []


def test_dict_windows_requires_rgb_channels():
    batch = default_collate([_neckflix_sample(channels=("R", "G", "I"))])
    with pytest.raises(ValueError, match=r"missing \['B'\]"):
        list(_dict_windows(batch))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_every_method_runs_over_a_dict_batch(method):
    batch = default_collate([_neckflix_sample(t=256), _neckflix_sample(t=256)])
    for trace, references, _ in _dict_windows(batch):
        bvp = estimate_bvp(method, trace, FS)
        assert np.isfinite(bvp).all()
        for reference in references.values():
            usable = min(len(bvp), len(reference))
            gt, pred, snr, macc = calculate_metric_per_video(
                bvp[:usable], reference[:usable], diff_flag=False, fs=FS,
                hr_method='FFT')
            assert np.isfinite([gt, pred, snr, macc]).all()


def test_green_is_the_green_channel_mean():
    clip = pulsatile_clip(n_frames=32)
    assert np.allclose(GREEN(clip), clip.mean(axis=(1, 2))[:, 1])


def test_fft_hr_of_a_pure_tone_is_its_frequency():
    """Guard the premise the HR assertions above rely on."""
    t = np.arange(600) / FS
    assert abs(_calculate_fft_hr(np.sin(2 * np.pi * 1.2 * t), fs=FS) - 72) < 2


# --- the full predictor, both contracts -----------------------------------
class _Cfg(dict):
    """Minimal attribute-style config stand-in for the predictor."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _predictor_config(metrics=("MAE", "RMSE", "MACC")):
    return _Cfg(
        MODE="unsupervised_method",
        DATA=_Cfg(DATASET="Neckflix"),
        INTERFACE=_Cfg(FS=FS),
        TEST=_Cfg(METRICS=list(metrics),
                  EVALUATION_METHOD="FFT",
                  EVALUATION_WINDOW=_Cfg(USE_SMALLER_WINDOW=False,
                                         WINDOW_SIZE=10)),
    )


def test_predictor_reports_one_row_per_signal_over_dict_batches():
    from unsupervised_methods.unsupervised_predictor import unsupervised_predict
    batches = [default_collate([_neckflix_sample(signals=("ABP", "CVP"),
                                                 present=("ABP", "CVP"), t=256)])]
    report = unsupervised_predict(_predictor_config(), {"unsupervised": batches}, "POS")
    assert set(report) == {"ABP", "CVP"}
    assert all(row["n"] > 0 for row in report.values())


def test_predictor_rejects_a_legacy_tuple_batch():
    """The tuple path died with main.py; a clear TypeError beats silent nonsense."""
    from unsupervised_methods.unsupervised_predictor import unsupervised_predict
    clip = pulsatile_clip(n_frames=256, hw=(5, 4))
    batches = [(torch.from_numpy(clip).float().unsqueeze(0),
                torch.randn(1, 256), ["subject1"], torch.tensor([0]))]
    with pytest.raises(TypeError, match="batch-dict contract"):
        unsupervised_predict(_predictor_config(), {"unsupervised": batches}, "CHROM")


def test_predict_many_matches_running_each_method_alone():
    """The one-pass path must not change any number."""
    from unsupervised_methods.unsupervised_predictor import (
        unsupervised_predict, unsupervised_predict_many,
    )
    sample = _neckflix_sample(signals=("ABP",), present=("ABP",), t=256)
    config = _predictor_config()
    methods = ["POS", "CHROM", "GREEN"]
    together = unsupervised_predict_many(
        config, {"unsupervised": [default_collate([sample])]}, methods)
    for method in methods:
        alone = unsupervised_predict(
            config, {"unsupervised": [default_collate([sample])]}, method)
        assert together[method]["ABP"] == pytest.approx(alone["ABP"])


def test_predictor_rejects_a_non_unsupervised_mode():
    from unsupervised_methods.unsupervised_predictor import unsupervised_predict
    config = _predictor_config()
    config["MODE"] = "train_and_test"
    batches = [default_collate([_neckflix_sample(t=256)])]
    with pytest.raises(ValueError, match="only supports unsupervised_method"):
        unsupervised_predict(config, {"unsupervised": batches}, "GREEN")


# --- the loaders' pseudo-PPG path -----------------------------------------
def _pos_signal_as_the_loaders_had_it(frames, fs, WinSec=1.6):
    """The POS block both loaders carried inline, with np.mat -> np.asmatrix."""
    import math

    from unsupervised_methods.methods import POS_WANG as pos_module
    RGB = pos_module._process_video(frames)
    n = RGB.shape[0]
    H = np.zeros((1, n))
    length = math.ceil(WinSec * fs)
    for end in range(n):
        start = end - length
        if start >= 0:
            Cn = np.asmatrix(np.true_divide(RGB[start:end, :],
                                            np.mean(RGB[start:end, :], axis=0))).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            h = h - np.mean(h)
            H[0, start:end] = H[0, start:end] + h[0]
    return np.asarray(np.transpose(utils.detrend(np.asmatrix(H).H, 100)))[0]


@pytest.mark.parametrize("n_frames", [150, 300, 401])
def test_shared_pos_signal_matches_the_loaders_inline_version(n_frames):
    from unsupervised_methods.methods.POS_WANG import pos_signal
    rng = np.random.default_rng(n_frames)
    frames = rng.uniform(20, 230, (n_frames, 7, 5, 3))
    assert np.allclose(pos_signal(frames, FS),
                       _pos_signal_as_the_loaders_had_it(frames, FS), atol=1e-10)


def test_pos_signal_is_the_unfiltered_half_of_pos_wang():
    from scipy import signal as scipy_signal

    from unsupervised_methods.methods.POS_WANG import POS_WANG as pos_filtered
    from unsupervised_methods.methods.POS_WANG import pos_signal
    frames = pulsatile_clip(n_frames=256)
    b, a = scipy_signal.butter(1, [0.75 / FS * 2, 3 / FS * 2], btype='bandpass')
    expected = scipy_signal.filtfilt(b, a, pos_signal(frames, FS).astype(np.double))
    assert np.allclose(pos_filtered(frames, FS), expected)


def test_pos_signal_handles_a_clip_shorter_than_its_window():
    from unsupervised_methods.methods.POS_WANG import pos_signal
    out = pos_signal(pulsatile_clip(n_frames=10), FS)
    assert out.shape == (10,) and np.all(out == 0)


