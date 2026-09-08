"""Windows over admitted stores, shaped to the interface.

One :class:`WindowedDataset` per dataset name, built from that dataset's
admitted stores (``src.datasets``) and the interface (``src.interface``).
Training instances draw one random window per (recording, perspective) per
epoch; the test instance strides through every recording at
``WINDOW_STRIDE``. Same class both sides, and the training instances are
wrapped in a ``ConcatDataset``.

A sample is one (store, perspective): perspectives are separate cameras, not
pixel-aligned, so they are never stacked. Reads the cache contract layout
(``docs/cache-contract.md``): ``{store}/{perspective}/{modality}/video/data``
``(C, T, H, W)`` uint8, ``{modality}/{trace}/data`` ``(T,)`` float64, and the
perspective's ``fps`` attr. What a store holds is read off the groups it
actually has, never off the writer's ``complete`` attr — that attr records
what a preprocessing run *asked for*, so a recording whose source is missing
a camera still lists it.

Per sample, in this order — preprocess what the store has, then pad:

1. read the native span of every modality present and resample it to ``FS``;
2. resize present frames to ``RESIZE``;
3. ``INPUT_PREPROCESSING`` on each present channel, ``LABEL_PREPROCESSING``
   on each present trace;
4. pad every demanded channel or trace the store lacks with zeros and a
   False mask.

Emitted (``default_collate`` prepends the batch axis)::

    {"frames":       {ch:  {prep: (T, H, W) float32}},  # frames["G"]["Raw"]
     "labels":       {sig: (T,) float32},
     "label_stats":  {sig: {stat: () float32}},         # physical units
     "channel_mask": {ch:  () bool},
     "label_mask":   {sig: () bool},
     "metadata":     {"dataset", "recording", "participant", "perspective": str,
                      "start_frame": int}}

Input preprocessing is per channel: each channel's window is standardised or
differenced with its own statistics, so a channel's block never depends on
which other channels the interface happens to demand.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from einops import rearrange

from dataset.data_loader.label_transforms import (STAT_NAMES, apply_norm,
                                                  finite_stats)
from neural_methods.frame_transforms import diff_normalized, resize_video, standardized
from neural_methods.signals import MODALITY_CHANNELS, TRACE_KEYS
from src.interface import InterfaceConfig

MODES = ("random", "strided")
#: Measured rates within this fraction of each other are the same nominal rate.
FPS_NOMINAL_TOLERANCE = 0.01

_PREPROCESS = {
    "Raw": lambda v: v,
    "Standardized": standardized,
    "DiffNormalized": diff_normalized,
}


# ---------------------------------------------------------------------------
# Resampling a store's native rate to FS
# ---------------------------------------------------------------------------
def same_nominal_rate(left: float, right: float) -> bool:
    return abs(left - right) <= FPS_NOMINAL_TOLERANCE * max(left, right)


@dataclass(frozen=True)
class WindowPlan:
    """How one window at ``FS`` maps onto a sample's native frames.

    ``span`` native frames are read from ``start``; ``offsets`` picks
    ``window_frames`` of them (identity or decimation when ``weights`` is
    None), or ``offsets`` is a ``(lo, hi)`` pair and ``weights`` the linear
    blend between them (upsampling, never duplication).
    """

    span: int
    stride: int
    offsets: object
    weights: np.ndarray | None

    def take(self, array: np.ndarray, axis: int) -> np.ndarray:
        if self.weights is None:
            if (self.offsets.shape[0] == array.shape[axis]
                    and self.offsets[-1] == self.offsets.shape[0] - 1):
                return array
            return np.take(array, self.offsets, axis=axis)
        lo, hi = self.offsets
        low = np.take(array, lo, axis=axis).astype(np.float32)
        high = np.take(array, hi, axis=axis).astype(np.float32)
        shape = [1] * low.ndim
        shape[axis] = -1
        return low + (high - low) * self.weights.reshape(shape)


def plan_window(native_fps: float, interface: InterfaceConfig, where: str) -> WindowPlan:
    target = interface.FS
    n = interface.window_frames
    if same_nominal_rate(native_fps, target):
        return WindowPlan(n, interface.stride_frames, np.arange(n), None)
    ratio = native_fps / target
    if ratio < 1.0:
        if interface.UPSAMPLING != "interpolate":
            raise ValueError(
                f"{where} was recorded at {native_fps} fps but the interface asks "
                f"for FS {target}; set UPSAMPLING: interpolate to opt into linear "
                f"upsampling, or lower FS")
        span = max(int(round(interface.WINDOW_SECONDS * native_fps)), 2)
        stride = max(int(round(interface.WINDOW_STRIDE * native_fps)), 1)
        positions = np.arange(n) * ratio
        lo = np.clip(np.floor(positions).astype(int), 0, span - 1)
        hi = np.clip(lo + 1, 0, span - 1)
        return WindowPlan(span, stride, (lo, hi), (positions - lo).astype(np.float32))
    span = max(int(round(interface.WINDOW_SECONDS * native_fps)), n)
    stride = max(int(round(interface.WINDOW_STRIDE * native_fps)), 1)
    offsets = np.clip(np.rint(np.arange(n) * ratio).astype(int), 0, span - 1)
    return WindowPlan(span, stride, offsets, None)


# ---------------------------------------------------------------------------
# Samples: one per (store, perspective), inspected once at construction
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    store: Path
    recording: str
    participant: str
    perspective: str
    modalities: list[str]          # present, e.g. ['rgb', 'ir']
    traces: dict[str, list[str]]   # canonical signal -> modalities carrying it
    frame_count: int
    hw: tuple[int, int]            # native (H, W) of the first modality
    plan: WindowPlan


def inspect_sample(store: Path, attrs: dict, perspective: str,
                   interface: InterfaceConfig) -> Sample:
    where = f"{store.name}/{perspective}"
    group = zarr.open_group(str(store), mode="r")[perspective]
    fps = group.attrs.get("fps")
    if fps is None:
        raise ValueError(f"{where}: perspective has no fps attr; regenerate the store")
    modalities = [m for m in MODALITY_CHANNELS
                  if m in group and "video" in group[m]]
    if not modalities:
        raise ValueError(f"{where}: no video modality present")
    traces: dict[str, list[str]] = {}
    counts, hw = [], None
    for m in modalities:
        video = group[m]["video"]["data"]
        counts.append(int(video.shape[1]))
        hw = hw or (int(video.shape[2]), int(video.shape[3]))
        for key, signal in TRACE_KEYS.items():
            if key in group[m]:
                traces.setdefault(signal, []).append(m)
    return Sample(
        store=store, recording=str(attrs.get("recording", store.stem)),
        participant=str(attrs.get("participant", "")), perspective=perspective,
        modalities=modalities, traces=traces, frame_count=min(counts), hw=hw,
        plan=plan_window(float(fps), interface, where))


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------
class WindowedDataset(torch.utils.data.Dataset):
    """Windows over one dataset's admitted stores, shaped to the interface."""

    def __init__(self, name: str, stores: dict[Path, dict],
                 interface: InterfaceConfig, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.name = name
        self.interface = interface
        self.mode = mode
        self.samples: list[Sample] = []
        for store, attrs in stores.items():
            root = zarr.open_group(str(store), mode="r")
            for perspective in sorted(root.group_keys()):
                sample = inspect_sample(store, attrs, str(perspective), interface)
                if sample.frame_count < sample.plan.span:
                    warnings.warn(
                        f"{name}: {store.name}/{perspective} has {sample.frame_count} "
                        f"frames, shorter than one window ({sample.plan.span}); skipped")
                    continue
                self.samples.append(sample)
        # (sample index, start) — start None = drawn at access time.
        self.windows: list[tuple[int, int | None]] = []
        for i, s in enumerate(self.samples):
            if mode == "random":
                self.windows.append((i, None))
            else:
                self.windows.extend(
                    (i, start) for start in
                    range(0, s.frame_count - s.plan.span + 1, s.plan.stride))
        self._warn_zero_coverage()

    # -- coverage -----------------------------------------------------------
    def channel_modality(self, channel: str) -> tuple[str, int] | None:
        for modality, channels in MODALITY_CHANNELS.items():
            if channels and channel in channels:
                return modality, channels.index(channel)
        return None

    def _warn_zero_coverage(self) -> None:
        where = f"{self.name} ({self.mode} split)"
        if not self.samples:
            warnings.warn(f"{where}: no samples at all")
            return
        for ch in self.interface.CHANNELS:
            source = self.channel_modality(ch)
            if source is None or not any(source[0] in s.modalities for s in self.samples):
                warnings.warn(f"{where}: no sample carries channel {ch}; "
                              f"it will be zeros with a False mask")
        for sig in self.interface.TRACES:
            if not any(sig in s.traces for s in self.samples):
                warnings.warn(f"{where}: no sample carries trace {sig}; "
                              f"it will be zeros with a False mask")

    def __len__(self) -> int:
        return len(self.windows)

    # -- one window ---------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        i, start = self.windows[idx]
        s = self.samples[i]
        if start is None:
            start = int(torch.randint(0, s.frame_count - s.plan.span + 1, (1,)).item())
        end = start + s.plan.span
        group = zarr.open_group(str(s.store), mode="r")[s.perspective]

        # 1. read + resample every present modality
        videos = {m: s.plan.take(np.asarray(group[m]["video"]["data"][:, start:end]), axis=1)
                  for m in s.modalities}
        # 2 + 3. resize and preprocess present channels; 4. pad the rest
        frames, channel_mask = {}, {}
        for ch in self.interface.CHANNELS:
            source = self.channel_modality(ch)
            present = source is not None and source[0] in videos
            if present:
                modality, index = source
                plane = torch.from_numpy(np.ascontiguousarray(videos[modality][index])).float()
                plane = rearrange(plane, "t h w -> 1 1 t h w")
                if self.interface.resizes:
                    plane = resize_video(plane, (self.interface.RESIZE.H, self.interface.RESIZE.W))
                frames[ch] = {prep: rearrange(_PREPROCESS[prep](plane), "1 1 t h w -> t h w")
                              for prep in self.interface.INPUT_PREPROCESSING}
            else:
                frames[ch] = {prep: torch.zeros((self.interface.window_frames, *self._pad_hw(s)))
                              for prep in self.interface.INPUT_PREPROCESSING}
            channel_mask[ch] = torch.tensor(present)

        labels, label_stats, label_mask = {}, {}, {}
        for sig in self.interface.TRACES:
            copies = [self._trace(group[m], sig, start, end, s.plan) for m in s.traces.get(sig, [])]
            trace = torch.from_numpy(_finite_mean(copies)).float() if copies else None
            present = trace is not None and bool(torch.isfinite(trace).any())
            if present:
                finite = torch.isfinite(trace)
                stats = finite_stats(trace)
                normed = apply_norm(trace, stats, self.interface.LABEL_PREPROCESSING[sig])
                labels[sig] = torch.where(finite, normed, trace.new_zeros(()))
            else:
                labels[sig] = torch.zeros(self.interface.window_frames)
                stats = {name: torch.zeros(()) for name in STAT_NAMES}
            label_stats[sig] = stats
            label_mask[sig] = torch.tensor(present)

        return {
            "frames": frames, "labels": labels, "label_stats": label_stats,
            "channel_mask": channel_mask, "label_mask": label_mask,
            "metadata": {"dataset": self.name, "recording": s.recording,
                         "participant": s.participant, "perspective": s.perspective,
                         "start_frame": start},
        }

    def _pad_hw(self, s: Sample) -> tuple[int, int]:
        if self.interface.resizes:
            return self.interface.RESIZE.H, self.interface.RESIZE.W
        return s.hw

    @staticmethod
    def _trace(modality, sig: str, start: int, end: int, plan: WindowPlan) -> np.ndarray:
        """One trace copy over the native span, NaN-padded if the trace is short."""
        data = modality[TRACE_KEYS_INVERSE[sig]]["data"]
        stop = min(end, int(data.shape[0]))
        sliced = np.asarray(data[start:stop], dtype=np.float64)
        if sliced.shape[0] < end - start:
            sliced = np.concatenate([sliced, np.full((end - start) - sliced.shape[0], np.nan)])
        if plan.weights is None:
            return plan.take(sliced, axis=0)
        lo, hi = plan.offsets
        return sliced[lo] + (sliced[hi] - sliced[lo]) * plan.weights.astype(np.float64)


TRACE_KEYS_INVERSE = {signal: key for key, signal in TRACE_KEYS.items()}


def _finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
    """Position-wise mean over finite values across trace copies; NaN where none."""
    stacked = np.stack(arrays)
    finite = np.isfinite(stacked)
    counts = finite.sum(axis=0)
    sums = np.where(finite, stacked, 0.0).sum(axis=0)
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
