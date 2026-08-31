"""The typed config schema: BASE merging, coercion, and unknown-key refusal."""
import textwrap

import pytest

from config import ConfigError, load_config

MINIMAL = """\
MODE: unsupervised_method
INTERFACE:
  FS: 30
  WINDOW_SECONDS: 2.0
  CHANNELS: [R, G, B]
  TRACES: [ABP, CVP]
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return str(path)


def test_minimal_config_loads_with_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "c.yaml", MINIMAL))
    assert cfg.MODE == "unsupervised_method"
    assert cfg.INTERFACE.CHANNELS == ["R", "G", "B"]
    assert cfg.INTERFACE.TRACES == ["ABP", "CVP"]
    assert cfg.INTERFACE.UPSAMPLING == "refuse"
    assert cfg.DATA.ALLOW_MISSING is True
    assert cfg.TEST.EVALUATION_METHOD == "FFT"
    assert "MAE" in cfg.TEST.METRICS


def test_int_where_a_float_lives_is_coerced_not_refused(tmp_path):
    """The yacs trap: STRIDE_SECONDS: 0 and FS: 30 must simply work."""
    cfg = load_config(write(tmp_path, "c.yaml", MINIMAL + """\
DATA:
  SPLITS:
    TRAIN: {STRIDE_SECONDS: 0}
"""))
    assert cfg.INTERFACE.FS == 30.0
    assert cfg.DATA.split("train").STRIDE_SECONDS == 0.0


def test_a_non_integer_rate_is_representable(tmp_path):
    """The cache's measured rate (29.9796...) must be writable as FS."""
    cfg = load_config(write(tmp_path, "c.yaml", MINIMAL.replace(
        "FS: 30", "FS: 29.97961373390558")))
    assert cfg.INTERFACE.FS == pytest.approx(29.97961373390558)


def test_unknown_keys_are_refused_with_the_path(tmp_path):
    with pytest.raises(ConfigError, match="WINDOW_FRAMES"):
        load_config(write(tmp_path, "c.yaml", """\
MODE: unsupervised_method
INTERFACE:
  FS: 30
  WINDOW_SECONDS: 2.0
  CHANNELS: [R]
  TRACES: [ABP]
  WINDOW_FRAMES: 64
"""))


def test_legacy_schema_keys_point_at_the_design_doc(tmp_path):
    with pytest.raises(ConfigError, match="pre-redesign"):
        load_config(write(tmp_path, "c.yaml", "TOOLBOX_MODE: train_and_test\n"))


def test_base_includes_deep_merge(tmp_path):
    write(tmp_path, "base.yaml", MINIMAL + """\
LOG_PATH: runs/base
TRAIN: {EPOCHS: 30, LR: 1e-3}
""")
    cfg = load_config(write(tmp_path, "smoke.yaml", """\
BASE: [base.yaml]
LOG_PATH: runs/smoke
TRAIN: {EPOCHS: 1}
INTERFACE: {RESIZE: 32}
"""))
    assert cfg.LOG_PATH == "runs/smoke"          # override wins
    assert cfg.TRAIN.EPOCHS == 1                 # override wins
    assert cfg.TRAIN.LR == 1e-3                  # inherited; YAML 1.2 float
    assert cfg.INTERFACE.CHANNELS == ["R", "G", "B"]   # inherited
    # RESIZE: 32 is the square shorthand for {H: 32, W: 32}
    assert (cfg.INTERFACE.RESIZE.H, cfg.INTERFACE.RESIZE.W) == (32, 32)


def test_per_model_blocks_become_attribute_namespaces(tmp_path):
    cfg = load_config(write(tmp_path, "c.yaml", MINIMAL + """\
MODEL:
  NAME: PhysFormer
  PHYSFORMER: {PATCH_SIZE: 4, DIM: 96}
"""))
    assert cfg.MODEL.PHYSFORMER.PATCH_SIZE == 4
    assert cfg.MODEL.PHYSFORMER.DIM == 96


def test_a_scalar_under_model_that_is_not_schema_is_refused(tmp_path):
    """A typoed MODEL key must not silently become a namespace."""
    with pytest.raises(ConfigError, match="MODEL.DROPRATE"):
        load_config(write(tmp_path, "c.yaml", MINIMAL + """\
MODEL: {NAME: PhysMamba, DROPRATE: 0.2}
"""))


def test_split_names_are_validated(tmp_path):
    with pytest.raises(ConfigError, match="EVAL"):
        load_config(write(tmp_path, "c.yaml", MINIMAL + """\
DATA:
  SPLITS:
    EVAL: {}
"""))


def test_fs_is_mandatory(tmp_path):
    with pytest.raises(ConfigError, match="INTERFACE.FS"):
        load_config(write(tmp_path, "c.yaml", MINIMAL.replace("  FS: 30\n", "")))
