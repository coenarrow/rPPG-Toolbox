"""Lazy torch datasets over external zarr caches.

The cache contract — store layout, root attrs, admission rules — is documented
in docs/architecture.md. Each cache is an external input written by its
dataset's preprocessor (for Neckflix: ghcr.io/coenarrow/neckflix >= 1.0.0);
this module never writes it. Per-dataset code is a ``channel_map`` subclass
plus a markdown cache spec beside it in this directory.
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import zarr

from dataset.data_loader.label_transforms import (
    STAT_NAMES, apply_norm, finite_stats, resolve_label_norms,
)

# Sentinel for root attrs absent from a store.
_MISSING = object()

#: How far two frame rates may differ and still count as the same *nominal*
#: rate. The preprocessor writes a measured rate (frames over elapsed time), not
#: the camera's nameplate one, so a nominally 30 fps Neckflix capture is written
#: as 29.9796 in one stream and 30.0 in another — the same camera, timed twice.
#: 1% separates that jitter from a genuinely different rate (30 vs 60, 15 vs 30),
#: which is the only thing worth refusing or decimating over.
FPS_NOMINAL_TOLERANCE = 0.01


def same_nominal_rate(left: float, right: float) -> bool:
    """True when two measured rates are the same nominal rate."""
    return abs(left - right) <= FPS_NOMINAL_TOLERANCE * max(left, right)


def _validate_filters(filters):
    """Reject overlapping include/exclude at construction.

    A lazy per-sample check could pass silently when no admitted sample
    carries the attribute, so this validates unconditionally upfront.
    """
    for attribute, spec in (filters or {}).items():
        overlap = set(spec.get("include", [])) & set(spec.get("exclude", []))
        if overlap:
            raise ValueError(
                f"Overlapping include/exclude for '{attribute}': {overlap}"
            )


def _sample(array: np.ndarray, offsets: np.ndarray, *, axis: int) -> np.ndarray:
    """Take ``offsets`` along ``axis``, skipping the copy when they are a no-op.

    Decimation is the exception, not the rule: most caches are already at the
    target rate, and there the offsets are a plain arange over the full span.
    """
    if offsets.shape[0] == array.shape[axis] and offsets[-1] == offsets.shape[0] - 1:
        return array
    return np.take(array, offsets, axis=axis)


def _sample_window(array: np.ndarray, offsets, weights, *, axis: int) -> np.ndarray:
    """Resample one window along ``axis`` per the window plan.

    ``weights is None`` is the integer path (identity or decimation, via
    :func:`_sample`); otherwise ``offsets`` is the plan's ``(lo, hi)`` pair and
    the result is the linear blend ``(1 - w) * a[lo] + w * a[hi]`` — the
    upsampling path, float-valued by construction.
    """
    if weights is None:
        return _sample(array, offsets, axis=axis)
    lo, hi = offsets
    low = np.take(array, lo, axis=axis).astype(np.float32)
    high = np.take(array, hi, axis=axis).astype(np.float32)
    shape = [1] * low.ndim
    shape[axis] = -1
    blend = weights.reshape(shape)
    return low + (high - low) * blend


class BaseZarrDataset(ABC, torch.utils.data.Dataset):
    """Abstract lazy dataset over external zarr stores.

    Subclasses provide only ``channel_map``. Construction is metadata-only:
    no pixel data is read until ``__getitem__``. Attribute filtering is
    generic — whatever root attrs a store carries can be filtered on.
    """

    MIN_TOOL_VERSION = (1, 0, 0)  # raw-frame cache format floor

    @property
    @abstractmethod
    def channel_map(self) -> dict[str, tuple[str, int]]:
        """Map channel names to ``(stream_group, channel_index)`` pairs."""
        ...

    def _resolve_streams(self, channels):
        """Map channel names through ``channel_map``; unknown -> ``None`` (zero-fill).

        A demanded channel this dataset can never provide is not an error: it
        is delivered as zeros with ``channel_mask=False`` — the convention the
        loader already uses for a stream a store happens to lack, extended to
        "this *dataset* has no such stream". That is what lets a checkpoint
        pretrained on RGBID run on an RGB-only dataset. One warning at
        construction; the per-sample masks carry the truth from there.
        """
        cmap = self.channel_map
        plan = [cmap.get(ch) for ch in channels]
        unknown = [ch for ch, entry in zip(channels, plan) if entry is None]
        if len(unknown) == len(list(channels)):
            raise ValueError(
                f"None of the demanded channels {list(channels)} exist in this "
                f"dataset (channel_map covers {list(cmap)}).")
        if unknown:
            warnings.warn(
                f"Channel(s) {unknown} are not provided by {type(self).__name__} "
                f"(channel_map covers {list(cmap)}); they will be delivered as "
                "zeros with channel_mask=False.")
        return plan

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cache_root = Path(cfg["cache_dir"])
        self.channels = cfg["channels"]
        self.labels = cfg["labels"]
        # The window is physical (duration + target rate); the frame count is
        # what those two derive, and only holds at the target rate. A store
        # recorded faster is decimated down to it, per sample.
        self.target_fps = float(cfg["target_fps"])
        self.window_seconds = float(cfg["window_seconds"])
        self.stride_seconds = float(cfg.get("stride_seconds") or self.window_seconds)
        self.window_size = cfg["window_size"]
        self.window_stride = cfg.get("window_stride", self.window_size)
        self.random_windows = cfg.get("random_windows", False)
        self.filters = cfg.get("filters", {})
        self.label_norms = resolve_label_norms(self.labels, cfg.get("label_norms"))
        # 'refuse' (default) keeps the historical behaviour: a store slower
        # than the target rate is an error. 'interpolate' opts into linear
        # frame/label blending — interpolation, never duplication, because
        # duplicated frames make DiffNormalized identically zero.
        self.upsampling = str(cfg.get("upsampling", "refuse"))
        if self.upsampling not in ("refuse", "interpolate"):
            raise ValueError(
                f"upsampling must be 'refuse' or 'interpolate', got "
                f"{self.upsampling!r}")
        self._native_fps_cache: dict[tuple[str, str], float] = {}
        _validate_filters(self.filters)

        self.stream_plan = self._resolve_streams(self.channels)
        self.required_streams = sorted(
            {s[0].lower() for s in self.stream_plan if s is not None})

        self.allow_missing = cfg.get("allow_missing", False)
        self.min_channels = cfg.get("min_channels", 1)
        self.min_labels = cfg.get("min_labels", 1)
        self.present_streams: dict[tuple[str, str], list[str]] = {}
        self.present_labels: dict[tuple[str, str], list[str]] = {}
        self.stream_hw: dict[str, tuple[int, int]] | None = None

        # Pipeline: scan -> discover -> filter -> window
        self.dataset_dict = self._scan_cache()
        self.samples = self.discover_samples()
        self._filter_by_attribute(self.filters)
        self._load_windows()
        self._warn_zero_coverage()

    def _scan_cache(self) -> dict:
        """Walk external zarr stores in ``cache_root`` into the recording dict.

        Admission gate (see spec): unreadable stores, stores whose root attrs
        lack ``complete is True`` (identity — JSON boolean true only), and
        stores whose ``tool_version`` parses below 1.0.0 (unparseable values
        count as 0) are skipped with a warning. Groups without stream
        sub-groups (root ``events/``) and stream groups without a ``video``
        child are ignored.
        """
        if not self.cache_root.exists():
            raise FileNotFoundError(
                f"Cache directory not found: {self.cache_root}. The zarr cache "
                "is an external input — generate it with the dataset's "
                "preprocessor first (Neckflix: ghcr.io/coenarrow/neckflix)."
            )

        cache_dict: dict = {}
        for store_path in sorted(self.cache_root.glob("*.zarr")):
            try:
                root = zarr.open_group(str(store_path), mode="r")
            except Exception as err:
                warnings.warn(
                    f"Skipping {store_path.name}: unreadable store ({err})"
                )
                continue
            attrs = dict(root.attrs)
            if attrs.get("complete") is not True:
                warnings.warn(
                    f"Skipping {store_path.name}: no 'complete: true' root attr "
                    "(partial or pre-contract preprocessor run); regenerate it."
                )
                continue
            version = str(attrs.get("tool_version", "0"))
            try:
                version_tuple = tuple(int(p) for p in version.split("."))
            except ValueError:
                version_tuple = (0,)
            if version_tuple < self.MIN_TOOL_VERSION:
                warnings.warn(
                    f"Skipping {store_path.name}: tool_version {version!r} predates "
                    "the raw-frame format (needs >= 1.0.0); regenerate it."
                )
                continue

            recording: dict = {"attrs": attrs}
            for perspective_name, perspective_group in root.groups():
                perspective: dict = {}
                for stream_name, stream_group in perspective_group.groups():
                    entries = {name for name, _ in stream_group.groups()}
                    if "video" not in entries:
                        continue  # non-stream group, e.g. a bare trace group
                    perspective[stream_name] = {name: {} for name in entries}
                if perspective:
                    recording[perspective_name] = perspective
            cache_dict[store_path.stem] = recording

        if not any(len(rec) > 1 for rec in cache_dict.values()):
            raise RuntimeError(
                f"No usable zarr stores under {self.cache_root} — every store "
                "was missing, incomplete, or pre-1.0.0. Regenerate the cache "
                "with the dataset's preprocessor."
            )
        return cache_dict

    def discover_samples(self) -> list[tuple[str, str]]:
        """Build the sample list, retaining partial samples when allowed.

        Records, for every admitted ``(recording, perspective)``, the streams
        and labels actually present (canonical order) in
        ``self.present_streams`` / ``self.present_labels``. With
        ``allow_missing`` a sample is kept when it has >= ``min_channels``
        present streams and >= ``min_labels`` present labels; otherwise the
        strict all-present rule applies.
        """
        samples: list[tuple[str, str]] = []
        for rec_name, rec_data in sorted(self.dataset_dict.items()):
            for perspective, perspective_data in sorted(rec_data.items()):
                if perspective == "attrs":
                    continue

                streams = perspective_data.keys()
                present_streams = [s for s in self.required_streams if s in streams]
                present_labels = [
                    lab for lab in self.labels
                    if any(lab.lower() in perspective_data[s] for s in present_streams)
                ]

                if self.allow_missing:
                    keep = (len(present_streams) >= self.min_channels
                            and len(present_labels) >= self.min_labels)
                else:
                    # Strict: every required stream present AND every stream in
                    # the perspective carries every configured label.
                    keep = (
                        len(present_streams) == len(self.required_streams)
                        and all(
                            all(lab.lower() in perspective_data[s] for lab in self.labels)
                            for s in streams
                        )
                    )
                if not keep:
                    continue

                self.present_streams[(rec_name, perspective)] = present_streams
                self.present_labels[(rec_name, perspective)] = present_labels
                samples.append((rec_name, perspective))
        return samples

    def _filter_by_attribute(self, filters=None) -> None:
        """Filter ``self.samples`` in place by attribute include/exclude.

        ``filters`` is ``{attribute: {"include": [...], "exclude": [...]}}``,
        keyed by whatever root attrs the stores carry: a value in ``exclude``
        drops the sample; a non-empty ``include`` whitelists. The
        pseudo-attribute ``"perspective"`` compares ``str()``-coerced values
        against the sample's perspective key. A sample whose store lacks a
        root attr fails any non-empty include and passes an exclude-only
        filter; one UserWarning per affected attribute is emitted after the
        pass. Overlap validation already happened upfront in ``__init__``.
        """
        filters = filters or {}
        missing: dict[str, set[str]] = {}
        filtered_samples = []
        for recording, perspective in self.samples:
            attrs = self.dataset_dict[recording]["attrs"]
            passed = True
            for attribute, spec in filters.items():
                include = spec.get("include", [])
                exclude = spec.get("exclude", [])
                if attribute == "perspective":
                    value = str(perspective)
                    include = [str(v) for v in include]
                    exclude = [str(v) for v in exclude]
                else:
                    value = attrs.get(attribute, _MISSING)
                    if value is _MISSING:
                        missing.setdefault(attribute, set()).add(recording)
                        if include:            # membership unprovable
                            passed = False
                            break
                        continue               # exclude-only: passes
                if value in exclude or (include and value not in include):
                    passed = False
                    break
            if passed:
                filtered_samples.append((recording, perspective))

        for attribute, stores in sorted(missing.items()):
            warnings.warn(
                f"Filter attribute '{attribute}' missing from store root attrs "
                f"of: {', '.join(sorted(stores))}",
                UserWarning,
            )
        self.samples = filtered_samples

    def attribute_values(self, attribute: str) -> list[str]:
        """Sorted unique values of a root attr (or 'perspective') over the
        current samples — the LOSO fold-enumeration primitive.

        Values are ``str()``-coerced before sorting; samples whose store lacks
        the attribute are silently skipped.
        """
        values: set[str] = set()
        for recording, perspective in self.samples:
            if attribute == "perspective":
                values.add(str(perspective))
                continue
            attrs = self.dataset_dict[recording]["attrs"]
            if attribute in attrs:
                values.add(str(attrs[attribute]))
        return sorted(values)

    def _sample_streams(self, recording_name: str, perspective: str) -> list[str]:
        """Per-sample present streams; full required set if sample unknown."""
        return self.present_streams.get(
            (recording_name, perspective), list(self.required_streams)
        )

    def _sample_traces(self, recording_name: str, perspective: str) -> list[str]:
        """Per-sample present labels lowercased; all labels if sample unknown."""
        labels = self.present_labels.get((recording_name, perspective), self.labels)
        return [lab.lower() for lab in labels]

    def _get_frame_count(self, recording_name: str, perspective: str) -> int:
        """Shortest aligned frame count for a sample, from ``num_frames`` attrs."""
        store_path = self.cache_root / f"{recording_name}.zarr"
        cam = zarr.open_group(str(store_path), mode="r")[perspective]
        length: int | None = None
        for stream_name in self._sample_streams(recording_name, perspective):
            try:
                video = cam[stream_name]["video"]
                n = int(video.attrs["num_frames"])
            except KeyError as err:
                raise RuntimeError(
                    f"{store_path.name}/{perspective}/{stream_name}: missing "
                    "video group or 'num_frames' attr; regenerate this store "
                    "with the dataset's preprocessor."
                ) from err
            length = n if length is None else min(length, n)
        assert length is not None, f"no streams for {recording_name}/{perspective}"
        return int(length)

    def _native_fps(self, recording_name: str, perspective: str) -> float:
        """The store's own frame rate for one sample, from the stream video attrs.

        The rate is a fact about the data, so it is read rather than
        configured. Streams within a perspective are index-aligned by the
        preprocessor, so they must agree on the *nominal* rate — but not on the
        measured one: the same 30 fps capture is written as 29.9796 by one
        stream and 30.0 by another, and 320 of the 332 stores in the current
        Neckflix cache disagree with themselves that way. Only a disagreement wider than the
        jitter band means the alignment the cache contract rests on is untrue,
        and that is refused loudly. The slowest stream is taken as the sample's
        rate, matching ``_get_frame_count``, which already takes the shortest.
        """
        key = (recording_name, str(perspective))
        if key in self._native_fps_cache:
            return self._native_fps_cache[key]

        store_path = self.cache_root / f"{recording_name}.zarr"
        cam = zarr.open_group(str(store_path), mode="r")[str(perspective)]
        rates: dict[str, float] = {}
        for stream_name in self._sample_streams(recording_name, str(perspective)):
            try:
                rates[stream_name] = float(cam[stream_name]["video"].attrs["fps"])
            except KeyError as err:
                raise RuntimeError(
                    f"{store_path.name}/{perspective}/{stream_name}: missing the "
                    "'fps' video attr, so the window duration cannot be converted "
                    "to frames; regenerate this store with the dataset's "
                    "preprocessor."
                ) from err
        slowest, fastest = min(rates.values()), max(rates.values())
        if not same_nominal_rate(slowest, fastest):
            raise RuntimeError(
                f"{store_path.name}/{perspective}: streams are at genuinely "
                f"different frame rates ({rates}), but the cache contract says "
                "their frames are index-aligned; regenerate this store."
            )
        native = slowest
        if (native < self.target_fps
                and not same_nominal_rate(native, self.target_fps)
                and self.upsampling != "interpolate"):
            raise ValueError(
                f"{store_path.name}/{perspective} was recorded at {native} fps but "
                f"the config asks for FS={self.target_fps}. Refusing to upsample "
                "by default: duplicated frames make DiffNormalized inputs "
                "identically zero — a silently dark motion branch. Either lower "
                "INTERFACE.FS to the cache's native rate, or opt into linear "
                "interpolation with INTERFACE.UPSAMPLING: interpolate."
            )
        self._native_fps_cache[key] = native
        return native

    def _window_plan(self, recording_name: str, perspective: str):
        """``(span, stride, offsets, weights)`` in the store's own frame units.

        ``span`` is how many native frames one ``WINDOW_SECONDS`` window covers
        and ``offsets`` picks ``window_size`` of them at the target rate.
        ``weights`` is ``None`` for the identity and decimation paths (integer
        nearest-index take); for an upsampled store (opted in via
        ``upsampling: interpolate``) ``offsets`` is a ``(lo, hi)`` index pair
        and ``weights`` the linear blend between them — interpolation, never
        frame duplication.

        A store at the target's *nominal* rate is taken to be at exactly the
        target rate: the sub-percent gap between a measured 29.9796 and a
        configured 30 is timing jitter, and resampling on it would jitter the
        window contents for nothing. So the whole current Neckflix cache takes
        the identity path — a plain contiguous slice, exactly as before — and
        resampling engages only across a genuinely different rate.

        Sampling (rather than filtering) is what keeps ``DATA_TYPE`` honest:
        the consumer-side transforms run on the emitted window, so a diff is
        taken between successive *sampled* frames and DiffNormalized keeps its
        1/target-fps meaning by construction.
        """
        native = self._native_fps(recording_name, perspective)
        if same_nominal_rate(native, self.target_fps):
            return (self.window_size, max(self.window_stride, 1),
                    np.arange(self.window_size), None)
        ratio = native / self.target_fps
        if ratio < 1.0:                    # slower store: linear interpolation
            span = max(int(round(self.window_seconds * native)), 2)
            stride = max(int(round(self.stride_seconds * native)), 1)
            positions = np.arange(self.window_size) * ratio
            lo = np.clip(np.floor(positions).astype(int), 0, span - 1)
            hi = np.clip(lo + 1, 0, span - 1)
            weights = (positions - lo).astype(np.float32)
            return span, stride, (lo, hi), weights
        span = max(int(round(self.window_seconds * native)), self.window_size)
        stride = max(int(round(self.stride_seconds * native)), 1)
        offsets = np.rint(np.arange(self.window_size) * ratio).astype(int)
        offsets = np.clip(offsets, 0, span - 1)
        return span, stride, offsets, None

    def _load_windows(self) -> None:
        """Build the window index from samples, in each store's own frame units.

        Strided mode emits ``range(0, frame_count - span + 1, stride)`` starts;
        random mode emits a single ``None``-start entry per sample (start chosen
        at access time). Samples shorter than one window are skipped in both
        modes. Starts stay in native frames so ``start_frame`` keeps naming a
        real index in the store it came from.
        """
        windows: list[tuple[str, str, int | None]] = []
        for recording_name, perspective in self.samples:
            frame_count = self._get_frame_count(recording_name, perspective)
            span, stride, _, _ = self._window_plan(recording_name, perspective)
            if frame_count < span:
                continue
            if self.random_windows:
                windows.append((recording_name, perspective, None))
                continue
            for start in range(0, frame_count - span + 1, stride):
                windows.append((recording_name, perspective, int(start)))
        self.windows = windows

    def _warn_zero_coverage(self) -> None:
        """One warning per demanded channel/trace no admitted sample carries.

        Benign at inference (zeros + a False mask, by design); in training it
        means the model is being taught to ignore that channel, or will never
        receive gradient for that trace. A warning rather than an error,
        deliberately: fine-tuning a wider pretrained model on narrower data is
        legitimate. Channels the dataset can never provide were already warned
        about at construction (``_resolve_streams``) and are skipped here.
        """
        if not self.samples:
            return
        streams_with_data: set[str] = set()
        for present in self.present_streams.values():
            streams_with_data.update(present)
        for ch_name, plan_entry in zip(self.channels, self.stream_plan):
            if plan_entry is not None and plan_entry[0].lower() not in streams_with_data:
                warnings.warn(
                    f"Channel {ch_name!r} is absent from every admitted sample: "
                    "it will be all zeros with channel_mask=False throughout. "
                    "Fine for inference with a wider checkpoint; in training it "
                    "teaches the model to ignore the channel.")
        labels_with_data: set[str] = set()
        for present in self.present_labels.values():
            labels_with_data.update(present)
        for label in self.labels:
            if label not in labels_with_data:
                warnings.warn(
                    f"Trace {label!r} is absent from every admitted sample: it "
                    "will carry label_mask=False throughout, so it is never "
                    "trained or scored on this data.")

    def __len__(self) -> int:
        return len(self.windows)

    def _ensure_stream_shapes(self) -> None:
        """Populate ``self.stream_hw`` (canonical (H, W) per required stream).

        Scans cached stores for each required stream's frame shape so absent
        channels can be zero-filled to a size consistent across samples.
        Freezes only once every required stream has a real shape; a stream
        never seen in any store keeps the fallback (first real shape found).
        """
        if getattr(self, "_stream_hw_complete", False):
            return
        found: dict[str, tuple[int, int]] = dict(getattr(self, "_stream_hw_found", {}))
        for rec_name, perspective in self.samples:
            store_path = self.cache_root / f"{rec_name}.zarr"
            if not store_path.exists():
                continue
            root = zarr.open_group(str(store_path), mode="r")
            try:
                cam = root[str(perspective)]
            except KeyError:
                continue
            for stream_name in self.required_streams:
                if stream_name in found:
                    continue
                try:
                    frames = cam[stream_name]["video"]["frames"]  # (C, T, H, W)
                    found[stream_name] = (int(frames.shape[-2]), int(frames.shape[-1]))
                except KeyError:
                    pass
        if not found:
            raise RuntimeError(
                "No cached streams found to infer fill shapes; check cache_dir stores"
            )
        self._stream_hw_found = found
        fallback = next(iter(found.values()))
        self.stream_hw = {s: found.get(s, fallback) for s in self.required_streams}
        # Zero planes for channels the dataset can never provide (a None entry
        # in the stream plan) are sized like the streams it can.
        self.fallback_hw = fallback
        self._stream_hw_complete = True

    def _window_trace(self, stream, trace_key: str, start: int, end: int,
                      offsets, weights) -> np.ndarray:
        """One trace copy over the native span, NaN-padded then resampled.

        Per-stream trailing-NaN trimming can leave a trace shorter than its
        stream's ``num_frames``; a window overlapping that tail yields a short
        slice, padded here so copies always align. Labels are index-aligned to
        frames in the cache, so they resample with the *same* plan the frames
        do and stay aligned afterwards (in float64, so a NaN neighbour keeps
        poisoning its blended positions — absorbed by the post-norm zeroing).
        """
        data = stream[trace_key]["data"]
        stop = min(end, int(data.shape[0]))
        sliced = np.asarray(data[start:stop], dtype=np.float64)
        if sliced.shape[0] < (end - start):
            pad = np.full((end - start) - sliced.shape[0], np.nan)
            sliced = np.concatenate([sliced, pad])
        if weights is None:
            return _sample(sliced, offsets, axis=0)
        lo, hi = offsets
        low, high = sliced[lo], sliced[hi]
        return low + (high - low) * weights.astype(np.float64)

    @staticmethod
    def _finite_mean(arrays: list[np.ndarray]) -> np.ndarray:
        """Position-wise mean over finite values across trace copies.

        Positions where every copy is non-finite stay NaN (absorbed by the
        post-norm NaN zeroing downstream). Warning-free equivalent of
        np.nanmean.
        """
        stacked = np.stack(arrays)                       # (n_copies, T)
        finite = np.isfinite(stacked)
        counts = finite.sum(axis=0)                      # (T,)
        sums = np.where(finite, stacked, 0.0).sum(axis=0)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    def __getitem__(self, idx: int) -> dict:
        """Load one window as the spec's nested dense dict (see class docstring).

        frames: {channel: (1, T, H, W) float32} raw pixels, zeros where the
        stream is absent; labels: {label: (T,) float32} normalised per
        that signal's mode in ``label_norms`` with finite-only stats; label_stats:
        physical-unit stats that normalised each window; channel_mask /
        label_mask: scalar bools; metadata: recording_id / camera_id /
        start_frame.
        """
        rec_name, camera_id, start = self.windows[idx]
        self._ensure_stream_shapes()

        present_streams = self._sample_streams(rec_name, str(camera_id))
        present_labels = set(
            self.present_labels.get((rec_name, str(camera_id)), self.labels)
        )
        span, _, offsets, weights = self._window_plan(rec_name, str(camera_id))

        if start is None:  # random window
            n_frames = self._get_frame_count(rec_name, camera_id)
            max_start = n_frames - span
            start = (
                int(torch.randint(0, max_start + 1, (1,)).item())
                if max_start > 0
                else 0
            )
        end = start + span

        store_path = self.cache_root / f"{rec_name}.zarr"
        root = zarr.open_group(str(store_path), mode="r")
        cam_group = root[str(camera_id)]

        # --- Load present streams' frames + label trace copies ---
        stream_frames: dict[str, np.ndarray] = {}
        label_accumulators: dict[str, list[np.ndarray]] = {
            name: [] for name in self.labels
        }
        for stream_name in present_streams:
            stream = cam_group[stream_name]
            video = stream["video"]["frames"]  # (C, T, H, W) raw frames
            # Read the contiguous native span (chunk-friendly), then resample
            # it to the target rate per the window plan.
            stream_frames[stream_name] = _sample_window(
                np.asarray(video[:, start:end]), offsets, weights, axis=1)
            for label_name in self.labels:
                trace_key = label_name.lower()
                if trace_key in stream:
                    label_accumulators[label_name].append(
                        self._window_trace(stream, trace_key, start, end,
                                           offsets, weights)
                    )

        # --- Dense frames: every channel, zeros where its stream is absent
        # (or where the dataset has no such stream at all: a None plan entry) ---
        frames: dict[str, torch.Tensor] = {}
        channel_mask: dict[str, torch.Tensor] = {}
        for ch_name, plan_entry in zip(self.channels, self.stream_plan):
            if plan_entry is None:
                h, w = self.fallback_hw
                frames[ch_name] = torch.zeros(
                    (1, self.window_size, h, w), dtype=torch.float32
                )
                channel_mask[ch_name] = torch.tensor(False, dtype=torch.bool)
                continue
            s_name, ch_idx = plan_entry
            present = s_name in stream_frames
            if present:
                arr = stream_frames[s_name][ch_idx][np.newaxis]  # (1, T, H, W)
                frames[ch_name] = torch.from_numpy(arr.copy()).float()
            else:
                h, w = self.stream_hw[s_name]
                frames[ch_name] = torch.zeros(
                    (1, self.window_size, h, w), dtype=torch.float32
                )
            channel_mask[ch_name] = torch.tensor(present, dtype=torch.bool)

        # --- Dense labels: finite-only stats + post-norm NaN zeroing (dev. 4) ---
        labels: dict[str, torch.Tensor] = {}
        label_stats: dict[str, dict[str, torch.Tensor]] = {}
        label_mask: dict[str, torch.Tensor] = {}
        for label_name in self.labels:
            arrays = label_accumulators[label_name]
            has_data = bool(arrays) and label_name in present_labels
            if has_data:
                raw = torch.from_numpy(self._finite_mean(arrays)).float()  # (T,)
                finite = torch.isfinite(raw)
                present = bool(finite.any())
            else:
                present = False
            if present:
                stats = finite_stats(raw)
                normed = torch.where(
                    finite, apply_norm(raw, stats, self.label_norms[label_name]),
                    raw.new_zeros(()),
                )
            else:
                normed = torch.zeros(self.window_size, dtype=torch.float32)
                stats = {
                    name: torch.zeros((), dtype=torch.float32)
                    for name in STAT_NAMES
                }
            labels[label_name] = normed
            label_stats[label_name] = stats
            label_mask[label_name] = torch.tensor(present, dtype=torch.bool)

        recording_id = root.attrs.get("recording", rec_name)

        return {
            "frames": frames,
            "labels": labels,
            "label_stats": label_stats,
            "channel_mask": channel_mask,
            "label_mask": label_mask,
            "metadata": {
                "recording_id": recording_id,
                "camera_id": camera_id,
                "start_frame": start,
            },
        }
