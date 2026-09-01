"""The per-signal composite loss for the multi-signal batch-dict contract.

Two things vary per signal, and they vary together (migration contract §3):

* an **absolute-class** signal (ABP, CVP) arrives in physical units and its
  *level* is part of the prediction, so it is scored with CCC plus L1 terms on
  the window mean and on the soft systolic/diastolic peaks — all in mmHg;
* a **shape-class** signal (PPG, ECG, RESP) arrives per-window z-scored and
  only its waveform means anything, so it is scored with negpearson.

So the loss is a *registry*: one spec per trace, naming a component family and
the weight of each component. That is also where the per-signal scale factors
live — raw ABP error is O(10 mmHg), CVP O(1 mmHg), and a CCC term is O(1) in
any units, so an unweighted sum would let ABP own every gradient. There are
deliberately no global or dataset-wide normalisation constants: the model
predicts physical units off an activation-free readout, and the weights are the
one place the units are reconciled.

Every component reduces **per sample** to ``(B,)``, which is what lets the
masking compose: each is averaged over the batch with the denominator clamped
to >= 1, so a signal no window in the batch carries contributes exactly 0 —
never NaN, never a sentinel.

Contract v2 splits the two halves: this module produces the *unweighted*
components (a model calls it inside its own forward, and the values ride the
batch as ``raw_losses``), and :func:`weight_losses` applies the config weights
and reduces them to the scalar to backpropagate — the mean over modules, as it
has always been over traces. Keeping them apart is what lets a run plot a
component's raw magnitude against its weighted contribution, which is how a
drowned or dominating term is spotted.

The statistics are derived from the predicted waveform itself (soft local
extrema + a temperature softmax, after ``PhysHydraLoss``), not from a second
head, so a waveform can never disagree with its own systolic and diastolic
values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neural_methods.signals import (
    ABSOLUTE, canonical_signal, signal_class, signal_scale, validate_traces,
)

_EPS = 1e-8

#: Softness of the local-extrema map and of the peak softmax, both as a
#: fraction of the window's own range — see :func:`soft_peak_stat`.
TAU = 0.2
TEMPERATURE = 0.03

#: Peak-search neighbourhood, as a fraction of the window: a local maximum has
#: to be the largest sample within +/- T/10, which at any plausible heart rate
#: is one beat's worth of context.
PEAK_WIDTH_FRACTION = 5


# --- per-sample components ----------------------------------------------
# Each maps (B, T) prediction + (B, T) label to a (B,) loss.
def mse(pred, label):
    """Pointwise squared error."""
    return ((pred - label) ** 2).mean(dim=-1)


def negpearson(pred, label):
    """``1 - r``: waveform shape only, blind to level and amplitude."""
    p = pred - pred.mean(dim=-1, keepdim=True)
    l = label - label.mean(dim=-1, keepdim=True)
    num = (p * l).sum(dim=-1)
    den = torch.sqrt((p ** 2).sum(dim=-1) * (l ** 2).sum(dim=-1) + _EPS)
    return 1.0 - num / den


def ccc(pred, label):
    """``1 - CCC``: the well-conditioned base term for an absolute signal.

    Dimensionless and O(1) whatever the units, yet unlike a correlation it
    penalises a wrong mean and a wrong amplitude — which is exactly the part of
    an absolute-class prediction that negpearson would throw away.
    """
    mx = pred.mean(dim=-1, keepdim=True)
    my = label.mean(dim=-1, keepdim=True)
    vx = ((pred - mx) ** 2).mean(dim=-1, keepdim=True)
    vy = ((label - my) ** 2).mean(dim=-1, keepdim=True)
    cov = ((pred - mx) * (label - my)).mean(dim=-1, keepdim=True)
    coefficient = (2 * cov) / (vx + vy + (mx - my) ** 2 + _EPS)
    return rearrange(1.0 - coefficient, "b 1 -> b")


def mean_l1(pred, label):
    """Absolute error of the window mean — mean arterial pressure, in mmHg."""
    return (pred.mean(dim=-1) - label.mean(dim=-1)).abs()


def soft_peak_stat(signal, scale, kind):
    """Differentiable mean local maximum (or minimum) of ``(B, T)``, as ``(B,)``.

    Two soft steps, both from ``PhysHydraLoss``: a local-extrema map (how close
    each sample is to the largest value in its neighbourhood) and a
    temperature-weighted average of the signal at those extrema. Together they
    approximate the mean systolic (or diastolic) value across the window's
    beats without a peak detector's non-differentiable argmax.

    ``scale`` makes both softness parameters unit-free. ``PhysHydraLoss`` used
    absolute ones, which only work on a normalised signal: at ``tau=0.2`` on a
    raw mmHg trace every sample within 0.2 mmHg of the local max counts as a
    peak and nothing else does, and a ``temperature=0.03`` softmax over
    logits of ~90/0.03 saturates to a hard argmax. Expressed as a fraction of
    the window's own range, the same constants behave identically on a
    z-scored ECG and on a 40 mmHg pulse pressure.
    """
    work = -signal if kind == "min" else signal
    padded = rearrange(work, "b t -> b 1 t")

    width = max(3, (work.shape[-1] // PEAK_WIDTH_FRACTION) | 1)   # odd, >= 3
    local_max = F.max_pool1d(padded, kernel_size=width, stride=1, padding=width // 2)
    local_max = rearrange(local_max, "b 1 t -> b t")

    # Distance below the local maximum, in units of the window's range.
    distance = ((local_max - work) / scale).clamp_min(-1.0)
    peak_map = torch.exp(-distance / TAU)

    logits = work / (TEMPERATURE * scale) + (peak_map + _EPS).log()
    weights = torch.softmax(logits, dim=-1)
    peak = (weights * work).sum(dim=-1)
    return -peak if kind == "min" else peak


def _peak_l1(pred, label, kind):
    """Absolute error of the soft systolic (``max``) or diastolic (``min``) value.

    The softness scale comes from the *label*, detached: both sides are then
    measured with the same ruler, and the ruler itself carries no gradient.
    """
    scale = (label.amax(dim=-1) - label.amin(dim=-1)).detach().clamp_min(_EPS)
    scale = rearrange(scale, "b -> b 1")
    return (soft_peak_stat(pred, scale, kind)
            - soft_peak_stat(label, scale, kind)).abs()


def peak_max_l1(pred, label):
    """Systolic error."""
    return _peak_l1(pred, label, "max")


def peak_min_l1(pred, label):
    """Diastolic error."""
    return _peak_l1(pred, label, "min")


def spectral(pred, label, fs, fmax):
    """L1 between band-limited log-magnitude spectra, shape only.

    Both spectra are mean-centred in the log domain, so this scores where the
    energy is rather than how much of it there is.
    """
    # rfft in float32: ComplexHalf is still experimental, and under AMP the
    # inputs arrive in bfloat16.
    spectrum_p = torch.fft.rfft(pred.float(), dim=-1).abs()
    spectrum_l = torch.fft.rfft(label.float(), dim=-1).abs()
    if fs and fmax:
        freqs = torch.fft.rfftfreq(pred.shape[-1], d=1.0 / fs).to(pred.device)
        keep = freqs <= fmax
        spectrum_p, spectrum_l = spectrum_p[..., keep], spectrum_l[..., keep]
    log_p = torch.log(spectrum_p + _EPS)
    log_l = torch.log(spectrum_l + _EPS)
    log_p = log_p - log_p.mean(dim=-1, keepdim=True)
    log_l = log_l - log_l.mean(dim=-1, keepdim=True)
    return (log_p - log_l).abs().mean(dim=-1)


#: Every component the loss registry understands. A new signal class adds an
#: entry here and a family below, not a redesign.
COMPONENTS = {
    'ccc': ccc,
    'mean': mean_l1,
    'max': peak_max_l1,
    'min': peak_min_l1,
    'negpearson': negpearson,
    'mse': mse,
    'spectral': spectral,
}

#: Components that need the frame rate (and so the configured ``FS``).
_NEEDS_FS = ('spectral',)

#: The component families a ``TYPE`` selects. ``WEIGHTS`` then overrides the
#: family's defaults, and may name any component in ``COMPONENTS`` — adding a
#: spectral term to an absolute-class signal is a weight, not a new type.
LOSS_TYPES = ('absolute', 'shape', 'mse')

#: Frequencies above this carry no cardiac or respiratory content worth
#: matching; the spectral term stops there when it is used.
DEFAULT_FMAX = 4.0


def default_weights(signal: str, loss_type: str) -> dict:
    """The component weights a ``TYPE`` implies for one signal.

    For the absolute family the L1 components are weighted ``1 / scale`` with
    the signal's own error scale from the registry (ABP 20 mmHg, CVP 5 mmHg),
    which is what puts a pressure error and a dimensionless CCC term on the
    same footing without normalising the data.
    """
    if loss_type == 'absolute':
        weight = 1.0 / signal_scale(signal)
        return {'ccc': 1.0, 'mean': weight, 'max': weight, 'min': weight}
    if loss_type == 'shape':
        return {'negpearson': 1.0}
    if loss_type == 'mse':
        return {'mse': 1.0}
    raise ValueError(f"Unknown loss TYPE {loss_type!r}; known: {list(LOSS_TYPES)}")


def default_loss_type(signal: str) -> str:
    """The family a signal's class implies when the config names none."""
    return 'absolute' if signal_class(signal) == ABSOLUTE else 'shape'


def resolve_loss_specs(traces, overrides=None) -> dict:
    """``{signal: {'type': str, 'weights': {component: float}}}`` for a trace list.

    ``overrides`` is the config's ``TRAIN.LOSS`` registry, in its YAML spelling
    (``{ABP: {TYPE: absolute, WEIGHTS: {CCC: 1.0, MEAN: 0.05}}}``). Omitting a
    signal — or the whole block — takes the default implied by its class.
    Naming a signal the run does not predict is an error rather than a silently
    ignored typo.
    """
    traces = validate_traces(traces)
    specs = {}
    for signal in traces:
        loss_type = default_loss_type(signal)
        specs[signal] = {'type': loss_type,
                         'weights': default_weights(signal, loss_type)}

    for name, spec in dict(overrides or {}).items():
        signal = canonical_signal(name)
        if signal not in specs:
            raise ValueError(
                f"TRAIN.LOSS names {name!r}, which is not in TRACES {traces}")
        spec = dict(spec or {})
        loss_type = str(spec.get('TYPE') or specs[signal]['type']).lower()
        weights = dict(default_weights(signal, loss_type))
        for component, weight in dict(spec.get('WEIGHTS') or {}).items():
            key = str(component).lower()
            if key not in COMPONENTS:
                raise ValueError(
                    f"TRAIN.LOSS[{signal}] weights unknown component "
                    f"{component!r}; known: {sorted(COMPONENTS)}")
            weights[key] = float(weight)
        specs[signal] = {'type': loss_type,
                         'weights': {k: w for k, w in weights.items() if w}}
    return specs


class PerSignalLoss(nn.Module):
    """Masked composite loss components per signal — contract v2's raw_losses.

    ``forward`` returns ``{signal: {component: () tensor}}``: every masked
    component value, **unweighted** and graph-attached, keyed by signal. Which
    term dominates is the first question debugging a multi-signal run raises,
    and a per-signal breakdown is what answers whether one signal is drowning
    the others — so the components, not a scalar, are the return value.

    The weights, and the single scalar to backpropagate, are
    :func:`weight_losses`'s job.
    """

    def __init__(self, traces, specs=None, fs=None, fmax=DEFAULT_FMAX):
        super().__init__()
        self.traces = validate_traces(traces)
        self.specs = resolve_loss_specs(self.traces, specs)
        self.fs = float(fs) if fs else None
        self.fmax = fmax
        needs_fs = [s for s, spec in self.specs.items()
                    if any(spec['weights'].get(c) for c in _NEEDS_FS)]
        if needs_fs and not self.fs:
            raise ValueError(
                f"The loss spec for {needs_fs} uses a spectral component, which "
                "needs the frame rate; set DATA.FS.")

    def _component(self, name, pred, label):
        if name in _NEEDS_FS:
            return COMPONENTS[name](pred, label, self.fs, self.fmax)
        return COMPONENTS[name](pred, label)

    def forward(self, preds, labels, label_mask):
        """Unweighted masked components per signal — contract v2's raw_losses.

        Reads    : preds, labels, label_mask (all keyed by signal)
        Returns  : {signal: {component: () tensor}}, graph-attached.
        Weighting is the trainer's job — see :func:`weight_losses`.
        """
        raw = {}
        for signal in self.traces:
            pred, label = preds[signal], labels[signal]
            mask = label_mask[signal].to(pred.dtype)                  # (B,)
            # Clamped denominator: a signal absent from every window in the
            # batch contributes exactly 0 instead of 0/0.
            denominator = mask.sum().clamp(min=1.0)
            raw[signal] = {
                component: (self._component(component, pred, label) * mask).sum()
                           / denominator
                for component in self.specs[signal]['weights']
            }
        return raw

    def extra_repr(self):
        return "\n".join(
            f"{signal}: {spec['type']} " + ", ".join(
                f"{c}={w:g}" for c, w in spec['weights'].items())
            for signal, spec in self.specs.items())


def weight_losses(raw, weights):
    """Apply config weights to a model's ``raw_losses`` dict.

    Reads    : ``raw`` = {module: {component: () tensor}} (unweighted,
               graph-attached), ``weights`` = {module: {component: float}};
               a component with no weight entry is weighted 1.0, which is how
               a model stage the config never mentions still contributes.
    Returns  : ``(total, weighted)``. ``total`` is the scalar to
               backpropagate — the mean over modules of each module's weighted
               component sum, which is exactly the old mean-over-signals when
               the modules are the signals. ``weighted`` mirrors ``raw`` as
               detached floats, plus a ``'total'`` per module, for logging.

    A module whose spec zeroed every component still contributes its zero to
    the mean, so the denominator is the module count either way — dropping it
    would silently rescale every other module's gradient.
    """
    zero = next((torch.zeros_like(value) for components in raw.values()
                 for value in components.values()), torch.zeros(()))
    module_totals, weighted = [], {}
    for module, components in raw.items():
        module_weights = weights.get(module, {})
        module_total, entries = zero, {}
        for component, value in components.items():
            term = module_weights.get(component, 1.0) * value
            entries[component] = float(term.detach())
            module_total = module_total + term
        entries['total'] = float(module_total.detach())
        weighted[module] = entries
        module_totals.append(module_total)
    return torch.stack(module_totals).mean(), weighted
