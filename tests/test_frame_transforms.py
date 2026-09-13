import torch

from src.frame_transforms import FRAME_TRANSFORMS, diff_normalized, resize_video, standardized


def test_standardized_is_zero_mean_unit_std():
    out = standardized(torch.rand(4, 5, 6) * 100 + 50)
    assert torch.allclose(out.mean(), torch.zeros(()), atol=1e-5)
    assert torch.allclose(out.std(), torch.ones(()), atol=1e-3)


def test_diff_normalized_formula_and_trailing_zero_frame():
    plane = torch.rand(5, 4, 4) * 200 + 10
    out = diff_normalized(plane)
    assert out.shape == plane.shape
    assert torch.equal(out[-1], torch.zeros_like(out[-1]))
    expected = (plane[1:] - plane[:-1]) / (plane[1:] + plane[:-1] + 1e-7)
    assert torch.allclose(out[:-1], expected / expected.std(), atol=1e-5)
    assert torch.isfinite(diff_normalized(torch.zeros(4, 2, 2))).all()


def test_resize_video_changes_only_spatial_dims():
    plane = torch.rand(4, 16, 16)
    assert resize_video(plane, (8, 6)).shape == (4, 8, 6)
    assert torch.equal(resize_video(plane, (16, 16)), plane)
    assert set(FRAME_TRANSFORMS) == {"Raw", "Standardized", "DiffNormalized"}
