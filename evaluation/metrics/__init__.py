"""Metric families. Which apply to a signal follows from its class, never from
a config key — a metric can then never go missing because a YAML forgot to
ask for it."""

from neural_methods.signals import is_absolute

#: Signal class -> the families computed for it.
FAMILIES = {
    "absolute": ("waveform", "rate", "clinical"),
    "shape": ("waveform", "rate"),
}


def families_for(signal) -> tuple:
    return FAMILIES["absolute" if is_absolute(signal) else "shape"]
