"""Train one model on a set of datasets, holding one participant out or none.

    uv run python scripts/train.py --datasets pure \\
        --test-participant-dataset pure --test-participant-id 01 \\
        --model physnet --interface configs/interfaces/physnet_interface.yaml \\
        --training configs/training/physnet_training.yaml

Fits the model by the recipe and writes the run directory —
``runs/<MODEL>_<DATASET>.<held-out participant or all>-..._<YYYYMMDDHHMM>``,
e.g. ``runs/PHYSMAMBA_PURE.all-NECKFLIX.24_202609091000``, or
``MODEL_FILE_NAME`` if the recipe names it — holding ``config.yaml`` (everything the run ran on), ``model.pt`` after every
epoch and ``losses.csv``. Nothing is inferred here: ``scripts/infer.py`` runs
the checkpoint over the held-out participant and ``scripts/eval.py`` scores
the records, each from the run directory alone.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                                # noqa: E402
from src.datasets import load_dataset_configs, load_stores       # noqa: E402
from src.distributed import init_runtime, shutdown               # noqa: E402
from src.experiment import (                                     # noqa: E402
    add_config_arguments, add_limit_argument, add_split_arguments,
    check_split_arguments, compile_config, limit_windows, print_model,
    print_runtime, print_setup, print_split, print_stores, run_name,
    split_stores, train_windows,
)
from src.interface import load_interface                         # noqa: E402
from src.models import build_model, load_model_config            # noqa: E402
from src.trainer import DEFAULT_RUNS_DIR, Trainer                # noqa: E402
from src.training import load_training                           # noqa: E402

SCRIPT = "scripts/train.py"


def build_parser() -> argparse.ArgumentParser:
    parser = add_config_arguments(argparse.ArgumentParser(
        description="Train one model, holding one participant out or none."))
    add_split_arguments(parser)
    parser.add_argument(
        "--runs-dir", metavar="PATH", default=DEFAULT_RUNS_DIR,
        help="where run directories land (default: runs/)")
    add_limit_argument(parser)
    return parser


def main(argv=None) -> Path:
    """Fit the run; returns its directory."""
    parser = build_parser()
    args = parser.parse_args(argv)
    check_split_arguments(parser, args)

    try:
        interface = load_interface(args.interface)
        model_config = load_model_config(args.model, interface)
        training = load_training(args.training)
        print_setup(interface, model_config, training)
        # Name the run now, before the process group exists: the name
        # carries the start minute, and every rank stamps it here, within
        # milliseconds of the launch, so they all land in one directory.
        run_dir = Path(args.runs_dir) / run_name(args, training)
        # Resolve device / precision / process group against this machine
        # first: under a launch that cannot run distributed, only rank 0
        # continues past this line.
        runtime = init_runtime(training)
        print_runtime(runtime, training)
        configs = load_dataset_configs(args.datasets)
        stores = load_stores(configs)
        print_stores(configs, stores)
        split = split_stores(stores, args.test_participant_dataset,
                             args.test_participant_id)
        print_split(split)
        train_dataset = limit_windows(train_windows(split, interface), args.limit_windows)
    except ValueError as err:          # ConfigError is a ValueError
        parser.error(str(err))

    # The model: one copy of the architecture per trace, widths from the
    # interface, dict in and dict out. The loss is the trainer's, not its.
    model = build_model(model_config, interface)
    print_model(model_config, model)

    print(f"run: {run_dir}")
    config = compile_config(SCRIPT, argv, args, interface, model_config, training,
                            runtime, configs, split, run_dir)
    try:
        trainer = Trainer(model, interface, training, runtime, run_dir, config)
    except ConfigError as err:
        parser.error(str(err))
    try:
        trainer.fit(train_dataset)
    finally:
        shutdown(runtime)
    return run_dir


if __name__ == "__main__":
    main()
