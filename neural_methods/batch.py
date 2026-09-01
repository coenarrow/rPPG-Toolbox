"""The Neckflix batch dict: key names and the einops moves in and out of it.

The zarr loader emits nested dicts keyed by canonical channel and signal names
(the contract is documented in ``docs/architecture.md``), and
those dicts travel unchanged through models, losses and evaluation. This module
is the single place that knows the key names and the exactly two shape moves the
contract needs: packing ``frames`` into a backbone tensor, and splitting a
backbone tensor back into per-signal predictions.

Per-sample (dataset ``__getitem__``)::

    {"frames":       {ch:  (1, T, H, W) float32},   # raw pixel values
     "labels":       {sig: (T,)         float32},   # per-window normalised
     "label_stats":  {sig: {stat: ()    float32}},  # physical units
     "channel_mask": {ch:  ()           bool},
     "label_mask":   {sig: ()           bool},
     "metadata":     {"recording_id": str, "camera_id": str, "start_frame": int,
                      "attrs": {str: str}}}

Collated (``default_collate``) every tensor gains a leading batch axis, and the
metadata strings become lists. A model adds ``PREDICTIONS`` -> ``{sig: (B, T)}``
and returns the whole dict, so any tensor anywhere in the pipeline is
identifiable by its key.
"""

from einops import rearrange, reduce
import torch

# --- Contract keys -------------------------------------------------------
FRAMES = "frames"
LABELS = "labels"
LABEL_STATS = "label_stats"
CHANNEL_MASK = "channel_mask"
LABEL_MASK = "label_mask"
METADATA = "metadata"
PREDICTIONS = "predictions"
RAW_LOSSES = "raw_losses"    # {module: {component: () tensor}} — model-written, unweighted
LOSSES = "losses"            # same structure, config-weighted — trainer-written
LABEL_UNITS = "label_units"  # {signal: str}, from the v2 cache's units attrs

#: Keys the loader emits. ``PREDICTIONS`` is added by models, never by loaders.
LOADER_KEYS = (FRAMES, LABELS, LABEL_STATS, CHANNEL_MASK, LABEL_MASK, METADATA)

#: Metadata sub-keys.
RECORDING_ID = "recording_id"
CAMERA_ID = "camera_id"
START_FRAME = "start_frame"
#: The store's own scalar root attrs, stringified. Dataset-agnostic: whatever
#: attrs a store carries, not a fixed Neckflix key list. This is what the
#: evaluation hierarchy groups by (participant, posture, session, ...).
ATTRS = "attrs"


def is_batch_dict(obj) -> bool:
    """True for anything shaped like a loader batch (dict with ``frames``)."""
    return isinstance(obj, dict) and FRAMES in obj


def require_batch_dict(batch) -> dict:
    """Raise a contract-specific error rather than a bare ``KeyError``/``TypeError``."""
    if not is_batch_dict(batch):
        raise TypeError(
            f"Expected a Neckflix batch dict with a {FRAMES!r} key, got "
            f"{type(batch).__name__}"
            + (f" with keys {sorted(batch)}" if isinstance(batch, dict) else "")
        )
    return batch


def batch_size(batch) -> int:
    """Batch length, read off any collated frames tensor."""
    frames = require_batch_dict(batch)[FRAMES]
    any_channel = next(iter(frames.values()))
    return int(any_channel.shape[0])


# --- The two shape moves -------------------------------------------------
def stack_frames(frames: dict, channels) -> torch.Tensor:
    """Pack ``{channel: (B, 1, T, H, W)}`` into ``(B, C, T, H, W)`` in ``channels`` order.

    The per-channel singleton axis is the loader's, not a batch axis: each
    channel is one plane, so stacking and folding that axis away reproduces the
    channel dimension every backbone expects.

    ``channels`` is the *model's* list, and the model is the authority on what
    it consumes: a channel the model expects but this batch does not carry is
    zero-filled rather than refused, so an RGB+IR+depth checkpoint runs on
    RGB-only data — degraded, not crashed. That is the same convention the
    loader already uses for a stream a store happens to lack, and the same one
    ``channel_mask`` records; here there is no mask to write, because the
    absence is a property of the data the caller already knows about.
    """
    present = [ch for ch in channels if ch in frames]
    if not present:
        raise KeyError(
            f"Batch carries none of the model's channels {list(channels)}; "
            f"it carries {sorted(frames)}"
        )
    reference = frames[present[0]]
    planes = [frames.get(ch) for ch in channels]
    stacked = torch.stack(
        [torch.zeros_like(reference) if plane is None else plane for plane in planes],
        dim=0)
    return rearrange(stacked, "c b plane t h w -> b (c plane) t h w")


def unstack_frames(video: torch.Tensor, channels) -> dict:
    """Inverse of :func:`stack_frames`: ``(B, C, T, H, W)`` -> per-channel dict."""
    if video.shape[1] != len(channels):
        raise ValueError(
            f"Tensor has {video.shape[1]} channels but {len(channels)} names were given"
        )
    planes = rearrange(video, "b c t h w -> c b 1 t h w")
    return {ch: planes[i] for i, ch in enumerate(channels)}


def split_signals(raw: torch.Tensor, traces) -> dict:
    """Split a backbone's ``(B, S, T)`` output into ``{signal: (B, T)}``.

    A backbone that emits ``(B, T)`` for a single trace is accepted and lifted,
    so single-signal models need no special-casing.
    """
    if raw.dim() == 2:
        raw = rearrange(raw, "b t -> b 1 t")
    if raw.dim() != 3:
        raise ValueError(f"Expected backbone output (B, S, T), got shape {tuple(raw.shape)}")
    if raw.shape[1] != len(traces):
        raise ValueError(
            f"Backbone emitted {raw.shape[1]} signal(s) but {len(traces)} trace(s) "
            f"are configured: {list(traces)}"
        )
    return {sig: raw[:, i, :] for i, sig in enumerate(traces)}


def stack_signals(signals: dict, traces) -> torch.Tensor:
    """Inverse of :func:`split_signals`: ``{signal: (B, T)}`` -> ``(B, S, T)``."""
    missing = [sig for sig in traces if sig not in signals]
    if missing:
        raise KeyError(f"Missing signal(s) {missing}; dict carries {sorted(signals)}")
    return rearrange([signals[sig] for sig in traces], "s b t -> b s t")


# --- Traversal helpers ---------------------------------------------------
def map_tensors(obj, fn):
    """Apply ``fn`` to every tensor in a nested dict/list, preserving structure."""
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [map_tensors(v, fn) for v in obj]
        return type(obj)(mapped) if isinstance(obj, tuple) else mapped
    return obj


def move_to_device(batch, device, non_blocking: bool = False):
    """Move every tensor in the batch to ``device``; leave metadata strings alone."""
    return map_tensors(batch, lambda t: t.to(device, non_blocking=non_blocking))


def detach_to_cpu(batch):
    """Detached CPU copy of every tensor in the batch (for saving/metrics)."""
    return map_tensors(batch, lambda t: t.detach().cpu())


def _index_sample(obj, i, n):
    """One batch element of a collated value.

    Tensors are indexed on their batch axis. ``default_collate`` turns
    non-tensor per-sample fields (the metadata strings) into length-``n``
    lists, so those are indexed too; anything else is passed through.

    A **0-dim** tensor is passed through unindexed: after collation every
    genuinely per-sample tensor carries a batch axis, so a scalar is by
    construction batch-level, not per-sample — the loss components a model
    writes into ``raw_losses`` are one value for the whole batch, and every
    sample's view of them is that same value.
    """
    if torch.is_tensor(obj):
        return obj if obj.ndim == 0 else obj[i]
    if isinstance(obj, dict):
        return {k: _index_sample(v, i, n) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and len(obj) == n:
        return obj[i]
    return obj


def iter_samples(batch):
    """Yield per-sample views of a collated batch, one dict per batch element.

    The result has the *un-collated* shapes, so per-sample code (the
    unsupervised methods, output saving) reads exactly the keys and shapes the
    dataset produced.
    """
    require_batch_dict(batch)
    n = batch_size(batch)
    for i in range(n):
        yield _index_sample(batch, i, n)


def sample_name(sample_metadata) -> str:
    """Stable per-window identity: ``{recording}_cam{camera}``."""
    return f"{sample_metadata[RECORDING_ID]}_cam{sample_metadata[CAMERA_ID]}"


def _one_sample_video(frames: dict, channels) -> torch.Tensor:
    """``{channel: (1, T, H, W)}`` (one sample) -> ``(1, C, T, H, W)``."""
    return stack_frames({ch: v.unsqueeze(0) for ch, v in frames.items()}, channels)


def frames_to_rgb_video(frames: dict, channels=("R", "G", "B")):
    """``{channel: (1, T, H, W)}`` (one sample) -> ``(T, H, W, 3)`` float64 numpy.

    The layout the unsupervised methods want: channels-last, spatial planes
    intact, raw pixel values (they do their own normalisation).
    """
    return rearrange(_one_sample_video(frames, channels), "1 c t h w -> t h w c").double().numpy()


def frames_to_rgb_trace(frames: dict, channels=("R", "G", "B")):
    """``{channel: (1, T, H, W)}`` -> per-frame spatial mean ``(T, 3)`` float64.

    Every traditional rPPG method starts by collapsing each frame to its mean
    RGB, so reducing here — in float32, on the tensor — avoids materialising a
    float64 copy of the whole clip just to average it away.
    """
    video = _one_sample_video(frames, channels)
    return reduce(video, "1 c t h w -> t c", "mean").double().numpy()
