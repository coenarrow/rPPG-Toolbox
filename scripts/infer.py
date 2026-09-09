"""Run a trained model over one participant and record every strided window.

    uv run python scripts/infer.py runs/PHYSNET_PURE.01_202609091000
    uv run python scripts/infer.py runs/PHYSNET_PURE.01_202609091000 \\
        --test-participant-dataset pure --test-participant-id 02 \\
        --out runs/PHYSNET_PURE.01_202609091000/pure_02

Everything the run needs is in ``RUN_DIR/model.pt``: the checkpoint carries
the compiled config, so the interface, the model config, the training recipe
(device, precision, batch size, loader workers) and the datasets are rebuilt
from it through the same parsers the files went through. The participant
defaults to the one the run held out; the two flags name another, and
``--datasets`` reads the participant from a different cache than the run
trained on. The records land in ``RUN_DIR/test_records/`` (or ``--out``):
``meta.json``, ``windows.csv``, and per recording and camera one
``<TRACE>.csv`` with the time axis, the label, the mean and spread of the
overlapping window predictions and one column per window, all in physical
units (``src/records.py``).
"""

import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError                                # noqa: E402
from src.datasets import (                                       # noqa: E402
    hold_out_participant, load_dataset_configs, load_stores,
)
from src.distributed import init_runtime, shutdown               # noqa: E402
from src.experiment import (                                     # noqa: E402
    add_limit_argument, add_split_arguments, check_split_arguments,
    git_state, limit_windows, load_checkpoint, print_model, print_runtime,
    print_setup, print_stores, rebuild, test_windows,
)
from src.models import build_model                               # noqa: E402
from src.records import RECORDS_DIR, write_records               # noqa: E402
from src.trainer import Trainer                                  # noqa: E402

SCRIPT = "scripts/infer.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a trained model over one participant and record "
                    "every strided window.")
    parser.add_argument(
        "run_dir", metavar="RUN_DIR",
        help="a run directory written by scripts/train.py (holds model.pt)")
    parser.add_argument(
        "--datasets", nargs="+", metavar="NAME",
        help="dataset config name(s) to read the participant from "
             "(default: the datasets the run trained on)")
    add_split_arguments(parser)
    parser.add_argument(
        "--out", metavar="DIR",
        help=f"the records directory to write (default: RUN_DIR/{RECORDS_DIR})")
    add_limit_argument(parser)
    return parser


def main(argv=None) -> list:
    """Record the participant's windows; returns them (``[]`` off the main rank)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    check_split_arguments(parser, args)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / RECORDS_DIR

    try:
        checkpoint = load_checkpoint(run_dir)
        setup = rebuild(checkpoint["config"])
        interface, model_config, training = setup.interface, setup.model, setup.training
        print_setup(interface, model_config, training)
        runtime = init_runtime(training)
        print_runtime(runtime, training)
        configs = load_dataset_configs(args.datasets) if args.datasets else setup.datasets
        stores = load_stores(configs)
        print_stores(configs, stores)
        dataset = args.test_participant_dataset or setup.test_participant_dataset
        participant = args.test_participant_id or setup.test_participant_id
        if dataset is None:
            raise ConfigError(
                f"{run_dir} held no participant out; name one with "
                f"--test-participant-dataset and --test-participant-id")
        split = hold_out_participant(stores, dataset, participant)
        test_dataset = limit_windows(test_windows(split, interface), args.limit_windows)
    except ValueError as err:          # ConfigError is a ValueError
        parser.error(str(err))

    model = build_model(model_config, interface)
    print_model(model_config, model)
    print(f"checkpoint: {run_dir}; records to {out_dir}")
    try:
        # No compiled config: the trainer writes nothing during inference.
        trainer = Trainer(model, interface, training, runtime, run_dir)
    except ConfigError as err:
        parser.error(str(err))
    # After construction: the trainer seeds the readout biases, and the
    # checkpoint has to win over that seed.
    trainer.model.load_state_dict(checkpoint["model_state"])
    try:
        records = trainer.test(test_dataset)
        if runtime.is_main:
            meta = {"dataset": dataset, "participant": str(participant),
                    "run_dir": str(run_dir),
                    "command": shlex.join([SCRIPT, *(sys.argv[1:] if argv is None else argv)]),
                    "git": git_state()}
            write_records(records, out_dir, interface, meta)
            print(f"test: {len(records)} windows written to {out_dir}")
        return records
    finally:
        shutdown(runtime)


if __name__ == "__main__":
    main()
