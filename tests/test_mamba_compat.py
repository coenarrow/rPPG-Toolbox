"""Tests for the mamba_ssm compatibility layer (BiMamba, fallback, parallel scan)."""
import numpy as np
import pytest
import torch

from neural_methods.model import mamba_compat as mc
from neural_methods.model.mamba_compat import BiMamba, MambaRef, make_mamba

requires_mamba_ssm = pytest.mark.skipif(
    not mc.HAS_MAMBA_SSM, reason="mamba_ssm not installed on this platform")


def test_make_mamba_plain_returns_the_available_implementation():
    m = make_mamba(32)
    if mc.HAS_MAMBA_SSM:
        from mamba_ssm import Mamba
        assert isinstance(m, Mamba)
    else:
        assert isinstance(m, MambaRef)


def test_make_mamba_bimamba_returns_bimamba():
    m = make_mamba(32, bimamba=True)
    assert isinstance(m, BiMamba)


def test_make_mamba_drops_unknown_kwargs():
    # kwargs from old fork call-sites must not crash vanilla packages
    m = make_mamba(32, bimamba=True, use_fast_path=False, not_a_real_kwarg=1)
    assert isinstance(m, BiMamba)


def test_bimamba_shape_and_finite():
    m = BiMamba(48)
    x = torch.randn(2, 100, 48)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_bimamba_uses_both_directions():
    m = BiMamba(32)
    x = torch.randn(2, 64, 32)
    m(x).sum().backward()
    fwd_grads = [p.grad for p in m.fwd.parameters()]
    bwd_grads = [p.grad for p in m.bwd.parameters()]
    assert all(g is not None for g in fwd_grads)
    assert all(g is not None for g in bwd_grads)
    assert any(g.abs().sum() > 0 for g in fwd_grads)
    assert any(g.abs().sum() > 0 for g in bwd_grads)


def test_bimamba_backward_direction_matters():
    """The bwd block must actually see the reversed sequence: zeroing the
    fwd block's contribution, output at t must depend on inputs later than t."""
    torch.manual_seed(0)
    m = BiMamba(16)
    x = torch.randn(1, 32, 16)
    x2 = x.clone()
    x2[:, -1, :] += 10.0  # perturb the LAST timestep
    with torch.no_grad():
        d = (m(x) - m(x2)).abs().sum(dim=-1).squeeze(0)
    assert d[0] > 0, "earliest output should react to a last-timestep change (backward path)"


# --- pure-torch fallback -------------------------------------------------
def test_mamba_ref_shapes_and_gradients():
    torch.manual_seed(0)
    block = MambaRef(24, d_state=8, d_conv=4, expand=2)
    x = torch.randn(3, 40, 24, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is not None for p in block.parameters())


def test_mamba_ref_is_causal():
    """Perturbing frame t must leave every output before t untouched."""
    torch.manual_seed(0)
    block = MambaRef(16, d_state=8).eval()
    x = torch.randn(1, 24, 16)
    perturbed = x.clone()
    perturbed[:, 12:] += 5.0
    with torch.no_grad():
        delta = (block(x) - block(perturbed)).abs().sum(dim=-1).squeeze(0)
    assert torch.allclose(delta[:12], torch.zeros(12), atol=1e-5)
    assert delta[12:].sum() > 0


def test_mamba_ref_handles_non_power_of_two_lengths():
    block = MambaRef(8, d_state=4)
    for length in (1, 5, 17, 33):
        out = block(torch.randn(1, length, 8))
        assert out.shape == (1, length, 8)
        assert torch.isfinite(out).all()


def test_parallel_scan_matches_a_sequential_recurrence():
    """h_t = a_t h_{t-1} + x_t, the primitive both scan paths implement."""
    torch.manual_seed(0)
    a = torch.rand(2, 3, 21, 4) * 0.9 + 0.05     # non-power-of-2 length
    x = torch.randn(2, 3, 21, 4)
    expected = torch.zeros_like(x)
    state = torch.zeros(2, 3, 4)
    for t in range(x.shape[-2]):
        state = a[..., t, :] * state + x[..., t, :]
        expected[..., t, :] = state
    assert torch.allclose(mc._scan_nograd(a, x), expected, atol=1e-5)


def test_parallel_scan_backward_matches_autograd():
    torch.manual_seed(0)
    a = (torch.rand(1, 2, 12, 3) * 0.8 + 0.1)
    x = torch.randn(1, 2, 12, 3)

    def sequential(a_, x_):
        outs, state = [], torch.zeros(1, 2, 3)
        for t in range(x_.shape[-2]):
            state = a_[..., t, :] * state + x_[..., t, :]
            outs.append(state)
        return torch.stack(outs, dim=-2)

    a1, x1 = a.clone().requires_grad_(), x.clone().requires_grad_()
    a2, x2 = a.clone().requires_grad_(), x.clone().requires_grad_()
    weights = torch.randn(1, 2, 12, 3)
    (sequential(a1, x1) * weights).sum().backward()
    (mc._parallel_linear_scan(a2, x2) * weights).sum().backward()
    assert torch.allclose(a1.grad, a2.grad, atol=1e-4)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4)


@pytest.mark.skipif(mc._selective_scan_ref_orig is None, reason="mamba_ssm ops unavailable")
def test_parallel_scan_matches_reference():
    torch.manual_seed(0)
    Bb, Dd, Ll, Nn = 2, 16, 200, 8  # non-power-of-2 L
    u = torch.randn(Bb, Dd, Ll)
    delta = torch.randn(Bb, Dd, Ll) * 0.5
    A = -torch.exp(torch.randn(Dd, Nn) * 0.5)
    Bv = torch.randn(Bb, Nn, Ll)
    Cv = torch.randn(Bb, Nn, Ll)
    Dv = torch.randn(Dd)
    z = torch.randn(Bb, Dd, Ll)
    db = torch.randn(Dd)

    def mk():
        return [t.clone().requires_grad_() for t in (u, delta, A, Bv, Cv, z)]

    a1, a2 = mk(), mk()
    ref = mc._selective_scan_ref_orig(a1[0], a1[1], a1[2], a1[3], a1[4], Dv, a1[5],
                                      delta_bias=db, delta_softplus=True)
    par = mc.selective_scan_parallel(a2[0], a2[1], a2[2], a2[3], a2[4], Dv, a2[5],
                                     delta_bias=db, delta_softplus=True)
    assert (ref - par).abs().max().item() < 1e-4
    ref.sum().backward()
    par.sum().backward()
    for t1, t2 in zip(a1, a2):
        assert (t1.grad - t2.grad).abs().max().item() < 1e-3


# --- the real package, when it is installed ------------------------------
@requires_mamba_ssm
def test_portable_mamba_has_exactly_the_reference_parameters():
    """The CPU fallback runs the real module's own weights, so the two parameter
    sets must be identical -- not merely similar."""
    from mamba_ssm import Mamba
    real, ref = Mamba(64, d_state=16), MambaRef(64, d_state=16)
    assert {k: tuple(v.shape) for k, v in real.state_dict().items()} == \
           {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    ref.load_state_dict(real.state_dict())          # strict, both directions
    real.load_state_dict(ref.state_dict())


@requires_mamba_ssm
def test_portable_mamba_keeps_the_vanilla_identity():
    """Subclassing must not change what a checkpoint or an isinstance sees."""
    from mamba_ssm import Mamba
    block = mc.PortableMamba(48, d_state=8)
    assert isinstance(block, Mamba)
    assert sorted(block.state_dict()) == sorted(Mamba(48, d_state=8).state_dict())


@requires_mamba_ssm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_portable_mamba_cpu_path_matches_the_fused_kernels():
    """Same weights, same input, two execution paths -- same function."""
    torch.manual_seed(0)
    block = mc.PortableMamba(64, d_state=16).eval()
    x = torch.randn(2, 96, 64)
    with torch.no_grad():
        on_cpu = block(x)
        on_gpu = block.cuda()(x.cuda()).cpu()
    assert (on_cpu - on_gpu).abs().max() < 1e-4


def test_physhydra_forward():
    from neural_methods.model.PhysHydra import PhysHydra
    with torch.no_grad():
        out = PhysHydra(in_channels=3, out_signals=1, frames=32)(torch.randn(1, 3, 32, 32, 32))
    pred = out[0] if isinstance(out, (tuple, list)) else out
    assert pred.shape == (1, 1, 32)
    assert torch.isfinite(pred).all()
