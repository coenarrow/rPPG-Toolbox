"""Style C: S complete copies of a single-signal architecture, in parallel.

Contract v2's default multi-signal form
(docs/plans/2026-09-01-contract-v2-design.md, Part 2): each predicted signal
gets its own untouched copy of the published architecture — input widened to
the demanded channels, everything downstream true to the paper. Cost is
parameters, by design: S traces is S times the model.
"""
import torch
import torch.nn as nn

from neural_methods.model.DictModel import DictModel


class ParallelSignals(DictModel):
    """One single-trace copy per signal, presented as one DictModel.

    ``make_copy(trace)`` builds the copy for one signal: any object with
    ``forward_video((B, C_in, T, H, W)) -> (B, 1, T)`` and ``output_layers()``.

    A copy is driven through ``forward_video`` only — never through its own
    ``forward``/``predict`` — so the parent owns the frame transform and the
    criterion, and a copy's own transform and criterion are both dead. That is
    why the copies can share the parent's ``spec.transform`` instance safely
    (``FrameTransform`` holds no parameters), and why ``attach_loss`` reaching
    only the parent is correct rather than an oversight.
    """

    def __init__(self, make_copy, channels, traces, frame_transform=None,
                 fs=0.0):
        super().__init__(channels=channels, traces=traces,
                         frame_transform=frame_transform, fs=fs)
        self.copies = nn.ModuleList([make_copy(trace) for trace in self.traces])
        # Window constraints are per-architecture, so every copy agrees;
        # surface the first copy's so the build-time check sees them. Note a
        # copy that is a SignalDictWrapper reports what its backbone declares,
        # which is what makes this the architecture's constraint and not the
        # wrapper's default.
        first = self.copies[0]
        self.temporal_divisor = getattr(first, 'temporal_divisor', 1)
        self.temporal_length = getattr(first, 'temporal_length', None)

    def output_layers(self):
        """One readout per copy, in traces order.

        At S > 1 this is the per-signal shape ``init_output_bias`` already
        handles (its style-B branch); at S == 1 the single-element list lands
        in the widened branch, which is also correct because that readout has
        exactly one bias entry. Both are covered — do not "fix" either.
        """
        return [layer for copy in self.copies for layer in copy.output_layers()]

    def forward_video(self, video):
        """Reads: the transformed clip. Returns: (B, S, T), traces order."""
        return torch.cat([copy.forward_video(video) for copy in self.copies],
                         dim=1)

    def extra_repr(self):
        return f"{super().extra_repr()}, copies={len(self.copies)}"
