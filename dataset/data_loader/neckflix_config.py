"""Translate the experiment config into the zarr loader's plain-dict config.

``BaseZarrDataset`` deliberately takes a plain dict, with no schema coupling
(see the loader design spec). This module is the one place that knows how the
DATA / INTERFACE / MODEL schema (``config.py``) maps onto it: the ``DATA``
block (plus one ``SPLITS`` entry) decides which stores and windows
participate, and the ``INTERFACE`` block decides what the loader must deliver
— channels, traces, rate, window, label norms, upsampling policy.

It also owns the participant-id convention mismatch: the repo says ``P015`` on
the command line, while the store's ``participant`` root attr is the unprefixed
``"015"`` the preprocessor writes.
"""

import re

from dataset.data_loader.label_transforms import resolve_label_norms

_PARTICIPANT_PREFIX = re.compile(r"^[Pp](?=\d)")

#: How far ``seconds x fps`` may sit from a whole frame and still be snapped to
#: it. Wide enough to absorb a decimal spelling of an exact fraction
#: (``4.266667 x 30`` is 128.00001 frames, not 128), narrow enough that a
#: genuinely ambiguous value (``4.27 x 30 = 128.1``) is refused.
WINDOW_TOLERANCE_FRAMES = 0.01


def window_frames(seconds, fps, *, key="WINDOW_SECONDS") -> int:
    """``seconds x fps`` as an exact frame count, or a config error.

    The temporal contract is physical (contract §1): a window is a duration
    and a rate, never a bare frame count, because 150 frames means 5 s of
    physiology at 30 fps and 1 s at 150 fps — and 1 s cannot carry a heart
    rate. The frame count is therefore always derived, and a duration that
    does not land on a whole frame is a mistake worth naming rather than
    rounding away.
    """
    seconds, fps = float(seconds), float(fps)
    if fps <= 0:
        raise ValueError(
            f"INTERFACE.FS must be a positive target frame rate to derive {key}; got {fps}"
        )
    if seconds <= 0:
        raise ValueError(f"{key} must be positive, got {seconds}")
    exact = seconds * fps
    frames = round(exact)
    if frames < 1 or abs(exact - frames) > WINDOW_TOLERANCE_FRAMES:
        nearest = max(frames, 1) / fps
        raise ValueError(
            f"{key}={seconds} at FS={fps} is {exact:.4f} frames, which is not a "
            f"whole frame. Use {key}={nearest:.6f} for {max(frames, 1)} frames, "
            "or change FS."
        )
    return int(frames)


def normalise_participant(participant) -> str:
    """``'P015'`` / ``'015'`` / ``15`` -> ``'015'``, the store's own spelling.

    A bare integer is zero-padded to three digits because that is what the
    preprocessor writes; anything already non-numeric is passed through so an
    unusual id is filtered on verbatim rather than mangled.
    """
    text = str(participant).strip()
    text = _PARTICIPANT_PREFIX.sub("", text)
    return text.zfill(3) if text.isdigit() else text


def participant_filter(include=(), exclude=()) -> dict:
    """Build the ``participant`` filter spec, normalising every id."""
    return {
        "include": [normalise_participant(p) for p in include or ()],
        "exclude": [normalise_participant(p) for p in exclude or ()],
    }


def build_filters(data, split=None, *, include_participants=(),
                  exclude_participants=()) -> dict:
    """Attribute include/exclude filters from ``DATA`` (+ split overrides + LOSO).

    ``DATA.FILTERS`` maps store root attrs (or the ``perspective`` pseudo-attr)
    to include whitelists — whatever attrs the cache carries, no fixed key
    list. ``[]`` means "do not filter on this attribute". A split may override
    the whole mapping (``SPLITS.<X>.FILTERS``); ``None`` inherits.
    Participants stay a separate surface (``PARTICIPANTS`` / the LOSO
    arguments) because their ids are normalised; a ``participant`` key in
    ``FILTERS`` is refused rather than left to bypass that normalisation.
    """
    configured_filters = data.FILTERS
    configured_participants = list(data.PARTICIPANTS or [])
    if split is not None:
        if split.FILTERS is not None:
            configured_filters = split.FILTERS
        if split.PARTICIPANTS is not None:
            configured_participants = list(split.PARTICIPANTS)

    filters = {}
    for attribute, values in (configured_filters or {}).items():
        if str(attribute) == "participant":
            raise ValueError(
                "Filter participants with DATA.PARTICIPANTS or the "
                "participant arguments, not FILTERS.participant — those "
                "paths normalise ids (P015 -> 015); this one would not."
            )
        values = list(values or [])
        if values:
            filters[str(attribute)] = {"include": values, "exclude": []}

    include = list(include_participants or ()) or configured_participants
    exclude = list(exclude_participants or ())
    if include or exclude:
        filters["participant"] = participant_filter(include, exclude)
    return filters


def label_norms(interface) -> dict:
    """``{signal: mode}`` for the interface's traces.

    ``LABEL_NORM`` is a per-signal mapping and may be omitted entirely: each
    signal then takes its class default (absolute -> ``raw``, shape ->
    ``zscore``).
    """
    return resolve_label_norms(interface.TRACES, interface.LABEL_NORM)


def frame_size(interface):
    """``(H, W)`` the models should see, or ``None`` to keep the cache's own size."""
    height, width = int(interface.RESIZE.H), int(interface.RESIZE.W)
    return (height, width) if height > 0 and width > 0 else None


def zarr_config(config, split, *, include_participants=(),
                exclude_participants=(), random_windows=None) -> dict:
    """Full plain-dict config for ``NeckflixDataset`` from the experiment config.

    ``split`` is ``'train'`` / ``'valid'`` / ``'test'`` / ``'unsupervised'``
    (the last takes the TEST split policy). ``random_windows`` overrides the
    split's ``RANDOM_WINDOWS`` when given.

    The window is physical: ``WINDOW_SECONDS`` and ``FS`` travel down to the
    loader together with the frame count they derive, because the loader needs
    the duration to resample a store whose native rate differs from ``FS``.
    """
    data, interface = config.DATA, config.INTERFACE
    split_cfg = data.split(split)
    fps = float(interface.FS)
    window_seconds = float(interface.WINDOW_SECONDS)
    stride_seconds = float(split_cfg.STRIDE_SECONDS or 0)
    window_size = window_frames(window_seconds, fps)
    if stride_seconds:
        stride_size = window_frames(stride_seconds, fps, key="STRIDE_SECONDS")
    else:
        stride_seconds, stride_size = window_seconds, window_size
    return {
        "cache_dir": data.CACHED_PATH,
        "channels": list(interface.CHANNELS),
        "labels": list(interface.TRACES),
        "target_fps": fps,
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "window_size": window_size,
        "window_stride": stride_size,
        "random_windows": bool(split_cfg.RANDOM_WINDOWS if random_windows is None
                               else random_windows),
        "label_norms": label_norms(interface),
        "upsampling": str(interface.UPSAMPLING),
        "allow_missing": bool(data.ALLOW_MISSING),
        "min_channels": int(data.MIN_CHANNELS),
        "min_labels": int(data.MIN_LABELS),
        "filters": build_filters(
            data, split_cfg,
            include_participants=include_participants,
            exclude_participants=exclude_participants,
        ),
    }
