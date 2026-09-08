"""Discover recordings and probe their metadata. No decoding happens here.

FILL IN: ``discover_recordings`` and ``scan_recording``. Both are metadata
only -- open containers for their headers, read sidecar CSV/JSON -- so the
scan over a whole dataset takes seconds, not hours, and a corrupt recording
fails on its own rather than aborting discovery of the rest.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StreamInfo:
    """One video stream of one perspective: where it is and how long it is."""
    source: Path            # the file (or directory of frames) to decode
    num_frames: int         # frame count from metadata, used to size the write


@dataclass
class RecordingInfo:
    """Everything the scan learned about one recording."""
    name: str                                   # becomes ``{name}.zarr``
    attrs: dict                                 # root attrs; see ``scan_recording``
    perspectives: dict[str, dict[str, StreamInfo]] = field(default_factory=dict)
    #                ^ perspective  ^ modality -> stream
    source_resolution: list[int] | None = None  # [H, W] of the raw frames


def discover_recordings(input_dir: Path) -> list[str]:
    """Names of every recording under ``input_dir``. A listing, no probing.

    FILL IN. Return a sorted list of recording names; each becomes one
    ``{name}.zarr`` store. Raise ``FileNotFoundError`` if ``input_dir`` does
    not look like this dataset at all (wrong directory), so the CLI can say
    so and exit 1 instead of reporting "0 recordings found".
    """
    raise NotImplementedError("discover_recordings: list this dataset's recordings")


def scan_recording(input_dir: Path, name: str) -> RecordingInfo:
    """Probe one recording's streams and build its root attrs.

    FILL IN. The returned ``RecordingInfo`` must satisfy:

    * ``attrs["participant"]`` is a **string** -- any format the dataset
      uses, taken verbatim; never an int. This is the one root attr the
      contract requires. ``writer.init_store`` refuses anything else.
    * Every other attr is optional and free-form: whatever the dataset
      states about the recording (posture, session, sex, age, ...). Write
      the values as the dataset spells them -- downstream configs filter on
      them exactly as written; nothing translates.
    * ``perspectives`` maps a perspective name (``"1"``, ``"2"``, ...) to
      the modalities it carries, each modality name drawn from the contract
      vocabulary (``gr``, ``rgb``, ``ir``, ``depth``, ``t``). A perspective
      is a set of modalities whose pixels are physically aligned; a camera
      that is not aligned with another is its own perspective.
    * ``source_resolution`` is probed from the data, not hardcoded.

    Any stream a recording lacks is simply absent from ``perspectives``;
    absence is not an error. A recording whose metadata is unusable should
    raise: the CLI records it as a scan failure and carries on.
    """
    raise NotImplementedError("scan_recording: probe one recording's metadata")
