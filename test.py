import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_records = torch.load("runs/deepphys_neckflix_15/test_records.pt")
fs = test_records['fs']
ts = set()

traces = sorted(test_records['windows'][0]['labels'].keys())
window_length = len(test_records['windows'][0]['labels'][traces[0]])/fs

for window in test_records['windows']:
    pass

# update the timestamps
window = test_records['windows'][0]
start_time = (window['metadata']['start_frame']/fs).numpy()
stop_time = (start_time + window_length)
ts.update(np.arange(start_time,stop_time,1/fs))

