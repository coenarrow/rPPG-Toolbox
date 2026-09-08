"""Where a run executes: which device, how many processes, and what to do
when the machine cannot give what the recipe asked for.

The recipe asks for a device and a precision; the launcher may ask for
several processes. This module resolves those against the machine and
returns one :class:`Runtime`. Every downgrade is a warning, never silence:

* launched under ``torch.distributed.run`` but distributed training cannot
  work here (no backend, fewer GPUs than ranks, no accelerator at all, Apple
  MPS) -> rank 0 carries on as a single process, every other rank exits;
* ``DEVICE: gpu`` with no CUDA and no MPS -> CPU, and a non-float32
  precision drops to float32 with it;
* CUDA without NCCL (Windows) -> the Gloo backend, which works but is slower.

Single-process runs never touch ``torch.distributed`` at all.
"""

import os
import platform
import sys
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist

from src.training import TrainingConfig


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    precision: str            # the precision actually used, maybe downgraded
    rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _launched_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _cuda_devices() -> int:
    """Visible CUDA devices. ``is_available()`` alone is not enough: with
    ``CUDA_VISIBLE_DEVICES=""`` it still answers True while the count is 0."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _resolve_device(training: TrainingConfig, local_rank: int) -> tuple[torch.device, str]:
    """The device and precision this process will actually use."""
    if training.DEVICE == "cpu":
        return torch.device("cpu"), training.PRECISION
    if _cuda_devices() > 0:
        return torch.device("cuda", local_rank), training.PRECISION
    if torch.backends.mps.is_available():
        return torch.device("mps"), training.PRECISION
    precision = training.PRECISION
    note = ""
    if precision != "float32":
        note = f"; PRECISION {precision} needs an accelerator, using float32"
        precision = "float32"
    warnings.warn(
        f"DEVICE is gpu but this {platform.system()} machine has neither CUDA "
        f"nor MPS; falling back to CPU{note}")
    return torch.device("cpu"), precision


def _distributed_blocker(device: torch.device, world_size: int,
                         training: TrainingConfig) -> str | None:
    """Why a multi-process launch cannot be a distributed run here, or None.

    A recipe that *asks* for CPU may run Gloo DDP across CPU processes (how a
    dev box exercises the distributed path); one that asked for a GPU and
    fell back to CPU may not — that is a degraded run, kept to one process.
    """
    if not dist.is_available():
        return "torch.distributed is not available in this build"
    if device.type == "mps":
        return "distributed training is not supported over Apple MPS"
    if device.type == "cuda" and _cuda_devices() < world_size:
        return f"only {_cuda_devices()} CUDA device(s) are visible"
    if device.type == "cpu" and training.DEVICE == "gpu":
        return "no accelerator is available"
    return None


def init_runtime(training: TrainingConfig) -> Runtime:
    """Resolve the recipe against this machine and the launcher's environment.

    Call once, first thing, in every process. Under a multi-process launch that
    cannot run distributed, the non-main processes exit here so exactly one
    run happens.
    """
    world_size = _launched_world_size()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device, precision = _resolve_device(training, local_rank if world_size > 1 else 0)

    if world_size <= 1:
        return Runtime(device=device, precision=precision)

    blocker = _distributed_blocker(device, world_size, training)
    if blocker is not None:
        if rank != 0:
            sys.exit(0)
        warnings.warn(
            f"{world_size} processes were launched but {blocker}; running as "
            f"a single process on {device} instead")
        return Runtime(device=device, precision=precision)

    if device.type == "cuda":
        if dist.is_nccl_available():
            backend = "nccl"
        else:
            backend = "gloo"
            warnings.warn(
                f"NCCL is not available on {platform.system()}; using the Gloo "
                f"backend for {world_size} CUDA processes, which is slower")
        torch.cuda.set_device(device)
    else:
        backend = "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return Runtime(device=device, precision=precision, rank=rank, world_size=world_size)


def all_reduce_sum(values: dict, runtime: Runtime) -> dict:
    """``{name: float}`` summed over ranks; a copy in a single-process run."""
    if not runtime.distributed:
        return dict(values)
    names = sorted(values)
    tensor = torch.tensor([values[n] for n in names], dtype=torch.float64,
                          device=runtime.device if runtime.device.type == "cuda" else "cpu")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return dict(zip(names, tensor.tolist()))


def gather_lists(items: list, runtime: Runtime) -> list:
    """Every rank's list, concatenated in rank order, on the main rank; ``[]``
    elsewhere. The list itself in a single-process run."""
    if not runtime.distributed:
        return items
    shards = [None] * runtime.world_size
    dist.all_gather_object(shards, items)
    if not runtime.is_main:
        return []
    return [item for shard in shards for item in shard]


def barrier(runtime: Runtime) -> None:
    if runtime.distributed:
        dist.barrier()


def shutdown(runtime: Runtime) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()
