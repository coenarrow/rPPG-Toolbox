import argparse
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import ConcatDataset, Subset

from src.evaluation.evaluate import evaluate
from src.config import ConfigError
from src.datasets import (
    Split, hold_out_participant, load_dataset_configs, load_stores,
    resolve_dataset_configs,
)
from src.distributed import Runtime, init_runtime, shutdown
from src.interface import DEFAULT_INTERFACE_PATH, load_interface
from src.models import build_model, load_model_config, resolve_model_config
from src.trainer import DEFAULT_RUNS_DIR, Trainer
from src.training import DEFAULT_TRAINING_PATH, load_training
from src.windows import WindowedDataset

REPO_ROOT = Path(__file__).resolve().parent


def limit_windows(dataset, limit: int):
    """Every k-th window so a smoke run touches the whole set, not its head."""
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    step = len(dataset) / limit
    return Subset(dataset, [int(i * step) for i in range(limit)])


def _git_state() -> dict:
    """``{commit, dirty}`` of the checkout, or ``None`` values outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def compile_config(args, argv, interface, model_config, training, runtime: Runtime,
                   datasets: dict, split: Split, run_dir: Path) -> dict:
    """Everything this run ran on, as one plain mapping.

    The four config sections (``datasets``, ``interface``, ``model``,
    ``training``) are exactly what their files loaded as after ``BASE``
    merging and validation, so each could be fed back to its loader;
    ``sources`` says which files those were. ``split`` lists the stores
    on each side of the hold-out, ``runtime`` the device and precision
    actually used (after any downgrade), ``git`` the code the run executed.
    Written to the run directory as ``config.yaml`` and carried inside the
    checkpoint.
    """
    return {
        "command": shlex.join(["run_experiment.py", *(sys.argv[1:] if argv is None else argv)]),
        "git": _git_state(),
        "run_dir": str(run_dir),
        "sources": {
            "datasets": {name: str(path.resolve())
                         for name, path in resolve_dataset_configs(args.datasets).items()},
            "interface": str(Path(args.interface).resolve()),
            "model": str(resolve_model_config(args.model).resolve()),
            "training": str(Path(args.training).resolve()),
        },
        "datasets": {name: asdict(cfg) for name, cfg in datasets.items()},
        "split": {
            "test_participant_dataset": args.test_participant_dataset,
            "test_participant_id": args.test_participant_id,
            "train": {name: sorted(p.name for p in kept) for name, kept in split.train.items()},
            "test": {name: sorted(p.name for p in kept) for name, kept in split.test.items()},
        },
        "interface": asdict(interface),
        "model": asdict(model_config),
        "training": asdict(training),
        "runtime": {
            "device": str(runtime.device),
            "precision": runtime.precision,
            "world_size": runtime.world_size,
        },
        "limit_windows": args.limit_windows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one experiment.")
    parser.add_argument(
        "--datasets", nargs="+", required=True, metavar="NAME",
        help="dataset config name(s), each resolved to "
             "configs/datasets/<NAME>.yaml (e.g. --datasets neckflix pure)")
    parser.add_argument(
        "--test-participant-dataset", metavar="NAME",
        help="which loaded dataset the test participant comes from")
    parser.add_argument(
        "--test-participant-id", metavar="ID",
        help="the participant held out of that dataset, exactly as its "
             "stores write it (e.g. '1' for Neckflix, '01' for PURE)")
    parser.add_argument(
        "--interface", metavar="PATH", default=DEFAULT_INTERFACE_PATH,
        help="the interface config (default: configs/interface.yaml)")
    parser.add_argument(
        "--model", required=True, metavar="NAME",
        help="model config name, resolved to configs/models/<NAME>.yaml "
             "(e.g. --model deepphys)")
    parser.add_argument(
        "--training", metavar="PATH", default=DEFAULT_TRAINING_PATH,
        help="the training recipe (default: configs/training.yaml)")
    parser.add_argument(
        "--runs-dir", metavar="PATH", default=DEFAULT_RUNS_DIR,
        help="where run directories land (default: runs/)")
    parser.add_argument(
        "--limit-windows", type=int, default=0, metavar="N",
        help="smoke runs: keep N evenly spaced windows of each split")
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.test_participant_dataset is None) != (args.test_participant_id is None):
        parser.error("--test-participant-dataset and --test-participant-id "
                     "go together")

    try:
        interface = load_interface(args.interface)
        print(f"interface: {interface.window_frames} frames per window at "
              f"{interface.FS} fps, test stride {interface.stride_frames} "
              f"frames, channels {interface.CHANNELS}, traces {interface.TRACES}")
        model_config = load_model_config(args.model, interface)
        print(f"model: {model_config}")
        training = load_training(args.training)
        print(f"training: {training}")
        # Resolve device / precision / process group against this machine
        # first: under a launch that cannot run distributed, only rank 0
        # continues past this line.
        runtime = init_runtime(training)
        if runtime.distributed:
            print(f"runtime: rank {runtime.rank} of {runtime.world_size} on "
                  f"{runtime.device}, global batch "
                  f"{training.BATCH_SIZE * runtime.world_size}")
        else:
            print(f"runtime: single process on {runtime.device}, "
                  f"precision {runtime.precision}")
        configs = load_dataset_configs(args.datasets)
        stores = load_stores(configs)
        for name, kept in stores.items():
            print(f"{name}: {len(kept)} stores admitted from "
                  f"{configs[name].CACHED_PATH}")
        if args.test_participant_dataset is None:
            return stores
        split = hold_out_participant(stores, args.test_participant_dataset,
                                     args.test_participant_id)
    except ValueError as err:          # ConfigError is a ValueError
        parser.error(str(err))

    for name, kept in split.train.items():
        print(f"train {name}: {len(kept)} stores")
    for name, kept in split.test.items():
        print(f"test  {name}: {len(kept)} stores "
              f"({', '.join(p.name for p in kept)})")

    # One WindowedDataset per dataset name; training draws one random window
    # per (recording, perspective), the test participant is strided.
    train_parts = [WindowedDataset(name, kept, interface, mode="random")
                   for name, kept in split.train.items() if kept]
    train_dataset = ConcatDataset(train_parts)
    (test_name, test_stores), = split.test.items()
    test_dataset = WindowedDataset(test_name, test_stores, interface, mode="strided")
    for part in train_parts:
        print(f"train {part.name}: {len(part)} windows (random, one per sample)")
    print(f"train total: {len(train_dataset)} windows")
    print(f"test  {test_dataset.name}: {len(test_dataset)} windows (strided)")

    # The model: one copy of the architecture per trace, widths from the
    # interface, dict in and dict out. The loss is the trainer's, not its.
    model = build_model(model_config, interface)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model_config.NAME} x {len(model.traces)} traces, "
          f"{model.in_channels} input channels, {n_params:,} parameters")

    # The run: <model>_<test dataset>_<test participant> unless the recipe
    # names it. Fit on the recipe, then record the test participant's windows.
    run_name = training.MODEL_FILE_NAME or \
        f"{args.model}_{args.test_participant_dataset}_{args.test_participant_id}"
    run_dir = Path(args.runs_dir) / run_name
    print(f"run: {run_dir}")
    train_dataset = limit_windows(train_dataset, args.limit_windows)
    test_dataset = limit_windows(test_dataset, args.limit_windows)
    config = compile_config(args, argv, interface, model_config, training, runtime,
                            configs, split, run_dir)
    try:
        trainer = Trainer(model, interface, training, runtime, run_dir, config)
    except ConfigError as err:
        parser.error(str(err))
    try:
        trainer.fit(train_dataset)
        records = trainer.test(test_dataset)
        if runtime.is_main:
            evaluate(records, run_dir, interface.FS)
        return records
    finally:
        shutdown(runtime)


if __name__ == "__main__":
    main()
