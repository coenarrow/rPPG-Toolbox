"""Load the trainer's test records from one or more run directories.

``src.trainer.Trainer.test`` writes ``test_records.pt`` as
``{"fs": float, "windows": [record, ...]}`` — one dict per test window, in
physical units, with the frame rate alongside so nothing downstream needs the
checkpoint. Pooling LOSO folds is loading several runs into one list; they
must share a frame rate or the waveform time axes would not agree.
"""

from pathlib import Path

import torch

RECORDS_NAME = "test_records.pt"


def load_records(run_dirs) -> tuple[list, float]:
    """``(windows, fs)`` over every run directory named, in order."""
    windows, fs = [], None
    for run_dir in run_dirs:
        path = Path(run_dir) / RECORDS_NAME
        if not path.is_file():
            raise FileNotFoundError(f"{run_dir} has no {RECORDS_NAME}")
        payload = torch.load(path, weights_only=False)
        if fs is None:
            fs = float(payload["fs"])
        elif float(payload["fs"]) != fs:
            raise ValueError(
                f"{path} was recorded at {payload['fs']} fps but earlier runs at "
                f"{fs}; pooling them would mix time bases")
        windows.extend(payload["windows"])
    return windows, fs
