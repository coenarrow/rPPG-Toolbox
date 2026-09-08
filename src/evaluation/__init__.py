"""The evaluation of a run's test records, with no config.

``evaluate`` scores levels (absolute signals) and heart rate (every cardiac
trace, fused and median) per window; ``rate`` is the spectral estimator;
``plots`` the figures; ``records`` the on-disk record format;
``post_process`` the upstream toolbox's detrend / bandpass / MACC helpers
the estimator and the unsupervised methods share.
"""
