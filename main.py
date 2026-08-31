"""Entry point for the zarr pipeline (formerly ``neckflix_main.py``).

Everything downstream of the loader speaks the nested batch dict (see
:mod:`neural_methods.batch`), so this script is short: load the
DATA / INTERFACE / MODEL config (``config.py``), build the splits, and hand
them to either the dict-contract trainer or the unsupervised predictor. The
legacy tuple-contract entry point this file replaces lives at the
``pre-overhaul`` tag.

Splits are participant-based (LOSO): the test split *includes* the named
participants and the train split *excludes* them. There is no percentage
slicing — the zarr cache has no notion of it.

Usage:
    Single process:
        uv run python main.py --config_file <config> --test_participants P015

    Single-node multi-GPU:
        uv run python -m torch.distributed.run --nproc_per_node=4 \
            main.py --config_file <config> --test_participants P015

    Unsupervised methods over the whole cache (no participant argument needed):
        uv run python main.py --config_file configs/neckflix/NECKFLIX_UNSUPERVISED.yaml
"""

import argparse
import os
# Must be set before importing torch: lets ops missing on Apple MPS
# (e.g. aten::max_pool3d_with_indices) fall back to CPU transparently.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from config import interface_diff, interface_from_payload, load_config
from dataset.data_loader.NeckflixLoader import NeckflixDataset
from dataset.data_loader.neckflix_config import normalise_participant, zarr_config
from neural_methods.trainer.MultiSignalTrainer import (
    MODEL_REGISTRY, MultiSignalTrainer, checkpoint_interface,
)
from unsupervised_methods.unsupervised_predictor import unsupervised_predict_many

#: DataLoader workers per process. Reading a Neckflix window means decompressing
#: a few hundred zarr frames, so this is usually the throughput limit; raise it
#: to match --cpus-per-task on a cluster.
DEFAULT_NUM_WORKERS = 4
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

UNSUPERVISED_METHODS = ("POS", "CHROM", "ICA", "GREEN", "LGI", "PBV", "OMIT")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    parser.add_argument('--config_file', required=True, type=str,
                        help="path to the config file")
    parser.add_argument('--test_participants', nargs='+', default=None,
                        help="participants held out for testing (LOSO); "
                             "'P015' and '015' are both accepted")
    parser.add_argument('--valid_participants', nargs='+', default=None,
                        help="participants held out for validation; excluded from training")
    parser.add_argument('--limit_windows', type=int, default=0,
                        help="evaluate/train on at most N evenly spaced windows per split "
                             "(0 = no limit). For smoke tests.")
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help="DataLoader workers per process "
                             f"(default {DEFAULT_NUM_WORKERS}); raise to match "
                             "--cpus-per-task on a cluster")
    return parser


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_dataset(config, split, *, include=(), exclude=(), random_windows=None,
                  limit=0):
    """One :class:`NeckflixDataset` for one split, plus LOSO ids."""
    dataset = NeckflixDataset(zarr_config(
        config, split,
        include_participants=include,
        exclude_participants=exclude,
        random_windows=random_windows,
    ))
    if limit and len(dataset) > limit:
        # Evenly spaced rather than the first N, so a subsample still spans
        # every recording rather than sitting inside the first one.
        indices = np.linspace(0, len(dataset) - 1, limit).round().astype(int).tolist()
        return Subset(dataset, sorted(set(indices)))
    return dataset


def make_loader(dataset, batch_size, *, shuffle, rank, world_size, drop_last, pin_memory,
                num_workers=DEFAULT_NUM_WORKERS):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        drop_last=drop_last,
    )


def build_data_loaders(config, args, rank, world_size, is_main):
    """All dataloaders the configured MODE needs, as a dict."""
    test_ids = args.test_participants or []
    valid_ids = args.valid_participants or []
    workers = getattr(args, "num_workers", DEFAULT_NUM_WORKERS)
    loaders = {}

    if config.MODE == "unsupervised_method":
        dataset = build_dataset(config, "unsupervised", include=test_ids,
                                random_windows=False, limit=args.limit_windows)
        _require_non_empty(dataset, "unsupervised")
        loaders["unsupervised"] = make_loader(
            dataset, config.TEST.BATCH_SIZE, shuffle=False, rank=rank,
            world_size=1, drop_last=False, pin_memory=False, num_workers=workers)
        return loaders

    if config.MODE == "train_and_test":
        train_dataset = build_dataset(
            config, "train", exclude=list(test_ids) + list(valid_ids),
            limit=args.limit_windows)
        _require_non_empty(train_dataset, "train")
        loaders["train"] = make_loader(
            train_dataset, config.TRAIN.BATCH_SIZE, shuffle=True, rank=rank,
            world_size=world_size, drop_last=True, pin_memory=True,
            num_workers=workers)

        if config.TEST.USE_LAST_EPOCH:
            loaders["valid"] = None
            if valid_ids and is_main:
                print("USE_LAST_EPOCH is set, so --valid_participants is ignored for "
                      "model selection (those participants are still excluded from "
                      "training).")
        elif not valid_ids:
            # Silently skipping validation here would leave best_epoch at 0 and
            # test the first checkpoint, which looks like a bad model rather than
            # a misconfiguration.
            raise ValueError(
                "TEST.USE_LAST_EPOCH is False but no --valid_participants were given, "
                "so there is nothing to select the best epoch with. Either pass "
                "--valid_participants, or set TEST.USE_LAST_EPOCH: True."
            )
        else:
            valid_dataset = build_dataset(config, "valid", include=valid_ids,
                                          random_windows=False, limit=args.limit_windows)
            _require_non_empty(valid_dataset, "valid")
            loaders["valid"] = make_loader(
                valid_dataset, config.TRAIN.BATCH_SIZE, shuffle=False, rank=rank,
                world_size=world_size, drop_last=False, pin_memory=True,
                num_workers=workers)

    # Test runs on rank 0 only, so it is never sharded.
    test_dataset = build_dataset(config, "test", include=test_ids,
                                 random_windows=False, limit=args.limit_windows)
    _require_non_empty(test_dataset, "test")
    loaders["test"] = make_loader(
        test_dataset, config.TEST.BATCH_SIZE, shuffle=False, rank=rank,
        world_size=1, drop_last=False, pin_memory=False, num_workers=workers)
    return loaders


def _require_non_empty(dataset, split):
    if len(dataset) == 0:
        raise ValueError(
            f"The {split} dataset is empty. Check DATA.CACHED_PATH, the "
            "participant arguments and the DATA.FILTERS attribute filters."
        )


# ---------------------------------------------------------------------------
# Config derivation
# ---------------------------------------------------------------------------
def adopt_checkpoint_interface(config, is_main=True):
    """In only_test mode, the checkpoint's interface is the authority.

    The checkpoint records the demand the model was trained against (rate,
    window, channels, traces, resize, DATA_TYPE, label norms). Adopting it
    before anything is built points the loaders at what the model actually
    expects — the config's INTERFACE block is only a guess about someone
    else's checkpoint. Differences are printed, not silently absorbed. A bare
    pre-redesign checkpoint carries no interface and leaves the config's in
    charge (the trainer warns at load time).
    """
    if config.MODE != "only_test" or not config.TEST.MODEL_PATH:
        return config
    payload = checkpoint_interface(config.TEST.MODEL_PATH)
    if payload is None:
        return config
    adopted = interface_from_payload(payload)
    differences = interface_diff(config.INTERFACE, adopted)
    if differences and is_main:
        print("Adopting the checkpoint's interface over the config's:")
        for line in differences:
            print("  " + line)
    config.INTERFACE = adopted
    return config


def apply_experiment_naming(config, args):
    """Name the experiment after what actually varies between Neckflix runs."""
    interface = config.INTERFACE
    channels = ''.join(interface.CHANNELS)
    traces = '-'.join(interface.TRACES)
    filters = ''.join(
        f"_{str(attr).upper()}-" + '-'.join(str(v) for v in values)
        for attr, values in sorted(config.DATA.FILTERS.items()) if values
    )
    exp_name = f"TRACES-{traces}{filters}_CHANNELS-{channels}" \
               f"_H-{interface.RESIZE.H}_W-{interface.RESIZE.W}"
    if args.test_participants:
        held_out = '_'.join(normalise_participant(p) for p in args.test_participants)
        exp_name = os.path.join(exp_name, f'tested_on_{held_out}')

    config.LOG.EXP_NAME = exp_name
    config.TEST.OUTPUT_SAVE_DIR = os.path.join(config.LOG.PATH, exp_name,
                                               'saved_test_outputs')
    config.UNSUPERVISED.OUTPUT_SAVE_DIR = os.path.join(config.LOG.PATH, exp_name,
                                                       'saved_outputs')
    config.MODEL.MODEL_DIR = os.path.join(config.LOG.PATH, exp_name,
                                          "PreTrainedModels")
    return config


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_supervised(config, loaders, rank, world_size):
    model_trainer = MultiSignalTrainer(config, loaders, rank=rank, world_size=world_size,
                                       debug=config.DEBUG)
    if config.MODE == "train_and_test":
        model_trainer.train(loaders)
    model_trainer.test(loaders)


def run_unsupervised(config, loaders, is_main=True):
    """Score every configured method. Rank 0 only: the pass is not sharded, so
    other ranks would duplicate the work and overwrite each other's plots."""
    if not is_main:
        return None
    if not config.UNSUPERVISED.METHODS:
        raise ValueError("Please set UNSUPERVISED.METHODS in the yaml!")
    unknown = [m for m in config.UNSUPERVISED.METHODS if m not in UNSUPERVISED_METHODS]
    if unknown:
        raise ValueError(f"Not supported unsupervised method(s): {unknown}. "
                         f"Available: {', '.join(UNSUPERVISED_METHODS)}")
    # One pass over the cache scores every configured method.
    return unsupervised_predict_many(config, loaders, config.UNSUPERVISED.METHODS)


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    config = load_config(args.config_file)

    is_distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_distributed:
        # The backend has to match where the model will actually live, not just
        # what the machine has: DEVICE: cpu with a GPU present must use gloo.
        on_cuda = torch.cuda.is_available() and 'cuda' in config.DEVICE
        init_process_group(backend="nccl" if on_cuda else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if 'cuda' in config.DEVICE:
            torch.cuda.set_device(local_rank)
    else:
        rank, world_size = 0, 1
    is_main = (rank == 0)

    config = adopt_checkpoint_interface(config, is_main)
    config = apply_experiment_naming(config, args)

    if is_main:
        print(f"Number of workers for data loading: {args.num_workers}")
        print(f"Running with {world_size} process(es); mode {config.MODE}")
        if config.MODE != "unsupervised_method" \
                and config.MODEL.NAME not in MODEL_REGISTRY:
            print(f"WARNING: model {config.MODEL.NAME!r} is not in the dict-contract "
                  f"registry {sorted(MODEL_REGISTRY)}")

    if config.DEBUG:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.autograd.set_detect_anomaly(True)
        if is_main:
            print("DEBUG MODE: Running with anomaly detection")

    if is_main:
        output_dir = (config.UNSUPERVISED.OUTPUT_SAVE_DIR
                      if config.MODE == "unsupervised_method"
                      else config.TEST.OUTPUT_SAVE_DIR)
        os.makedirs(output_dir, exist_ok=True)
        with open(args.config_file, 'r') as source, \
                open(os.path.join(output_dir, 'config.yaml'), 'w') as destination:
            destination.write(source.read())
        print(f"Saved a copy of the config file to: {output_dir}/config.yaml\n")

    if is_distributed:
        dist.barrier()

    loaders = build_data_loaders(config, args, rank, world_size, is_main)
    if is_main:
        for split, loader in loaders.items():
            if loader is not None:
                print(f"{split} dataset has {len(loader.dataset)} windows.")
    if is_distributed:
        dist.barrier()

    try:
        if config.MODE == "unsupervised_method":
            run_unsupervised(config, loaders, is_main)
        else:
            run_supervised(config, loaders, rank, world_size)
    finally:
        if is_distributed and dist.is_initialized():
            destroy_process_group()


if __name__ == "__main__":
    main()
