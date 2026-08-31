"""PURE dataset over its zarr cache.

Provides only the PURE channel map; all loading logic lives in
``BaseZarrDataset``. The cache is written by ``tools/cache_pure.py`` from the
raw PNG sequences; the store contract is documented in docs/architecture.md and
the raw-to-store mapping in ``PURE.md`` beside this file.

PURE is RGB-only, so a checkpoint whose ``INTERFACE.CHANNELS`` also demands I
or D still runs: those channels arrive zero-filled with ``channel_mask=False``,
which is what makes cross-dataset evaluation against Neckflix possible.
"""

from dataset.data_loader.zarr_dataset import BaseZarrDataset


class PUREDataset(BaseZarrDataset):
    """Dataset for PURE zarr stores."""

    _CHANNEL_MAP = {
        "R": ("rgb", 0),
        "G": ("rgb", 1),
        "B": ("rgb", 2),
    }

    @property
    def channel_map(self) -> dict[str, tuple[str, int]]:
        """PURE channel name to (stream_group, channel_index) mapping."""
        return self._CHANNEL_MAP
