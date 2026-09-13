"""``to_physical`` inverts the label normalisation exactly and leaves raw alone."""
import numpy as np
import torch

from src.trainer import to_physical


def _record():
    t = torch.arange(180, dtype=torch.float32) / 30
    abp = 100 + 20 * torch.sin(2 * np.pi * 1.2 * t)
    ecg = 5 * torch.sin(2 * np.pi * 1.2 * t)

    def stats(x):
        return {"mean": x.mean(), "std": x.std(), "min": x.amin(), "max": x.amax()}

    return {"predictions": {"ABP": abp, "ECG": ecg}, "labels": {"ABP": abp, "ECG": ecg},
            "label_stats": {"ABP": stats(abp), "ECG": stats(ecg)}}


def test_to_physical_inverts_zscore_and_leaves_raw():
    record = _record()
    stats = record["label_stats"]["ECG"]
    z = (record["labels"]["ECG"] - stats["mean"]) / stats["std"]
    normed = {**record, "labels": {**record["labels"], "ECG": z}}
    physical = to_physical(normed, {"ABP": "raw", "ECG": "zscore"})
    torch.testing.assert_close(physical["labels"]["ECG"], record["labels"]["ECG"],
                               rtol=1e-4, atol=1e-3)
    assert physical["labels"]["ABP"] is record["labels"]["ABP"]
    assert physical["label_stats"] is record["label_stats"]
