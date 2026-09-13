"""The evaluation of a run's records, with no config, interface file or
checkpoint — though torch still arrives transitively today.

One recording and camera at a time: ``recording.py`` (with ``beats.py``
and ``rate.py``) scores a folder's trace tables into beats, readings and
heart rates, written beside them. ``scripts/eval.py`` is the entry point;
``docs/evaluation.md`` the reference.
"""
