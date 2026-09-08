---
name: running-hpc-jobs
description: Use when running, training, preprocessing, testing, or benchmarking anything on the HPC cluster - submitting SLURM jobs, picking a partition or GPU type, writing .slurm scripts, requesting an interactive salloc session, launching multi-GPU distributed training, or diagnosing jobs stuck pending, OOM errors, port conflicts, and nproc_per_node mismatches.
---

# Running HPC Jobs

## The Iron Rule

**This repo is checked out on a login node. NEVER run computational tasks directly.**

Every training, inference, preprocessing, or benchmark run goes through SLURM — `sbatch` for
production, `salloc` for interactive debugging. No exceptions:

- Not "just a quick test"
- Not "only a few epochs"
- Not "just to see if it imports" (use `salloc`)
- Not preprocessing, which is CPU/IO heavy and will still get you killed —
  and not the first `uv run --project dataset/cachers/neckflix ...` either, which
  syncs a second 249 MB environment before it does any work

## GPU Resources

| Partition | Max GPUs | Resource flag | Use for |
|-----------|----------|---------------|---------|
| `gpu` | 2x V100 | `--gres=gpu:v100:N` | Development/testing (preferred) |
| `pophealth` | 4x A100 | `--gres=gpu:a100:N` | Dev fallback, production runs |
| `medical` | 4x H100 | `--gres=gpu:h100:N` | Dev last resort, primary for production |

### Choosing a partition for dev/test runs (<10 min expected)

Escalate only when the job is actually pending:

1. `gpu` with V100s. If pending →
2. `scancel`, resubmit to `pophealth` with A100s. If pending →
3. `scancel`, resubmit to `medical` with H100s. If pending →
4. Back to `gpu` and let it queue.

Test runs use **2 GPUs** — if it works on 2, assume it works on 3 and 4.

## Writing a SLURM Script

Scripts live in [.slurm_scripts/](../../../.slurm_scripts/) as `<Dataset>_<Model>_<Options>.slurm`.
Copy an existing script rather than writing from scratch —
`.slurm_scripts/Neckflix_PhysMamba_4GPU.slurm` is the canonical template.

```bash
#!/bin/bash
#SBATCH --job-name=Neckflix_PhysMamba_2GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                  # size to the job; NEVER --mem=0
#SBATCH --gres=gpu:v100:2
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_Neckflix_PhysMamba_2GPU.out
#SBATCH --error=logs/%j_Neckflix_PhysMamba_2GPU.err

mkdir -p logs
cd "/group/pgh004/carrow/repo/remote-physiology"
module load cuda

uv run python -m torch.distributed.run --nproc_per_node=2 \
    main.py --config_file configs/neckflix/NECKFLIX_PHYSMAMBA.yaml \
    --test_participants P015
```

Required in every script:

- **Log naming**: `logs/%j_<Model>_<Dataset>_<Options>.{out,err}`, plus `mkdir -p logs`
- **`module load cuda`** — GPU jobs fail without it
- **`uv run`** for all Python; never bare `python`
- **`--nproc_per_node` must equal the `--gres` GPU count**
- **`--mem`** set to the minimum the job needs (a 2-GPU dev run fits in 32G)

## Job Arrays: LOSO Cross-Validation

```bash
#SBATCH --partition=medical
#SBATCH --gres=gpu:h100:4
#SBATCH --array=1-57%1        # one job per participant, serialised

PORT=$((29500 + (SLURM_JOB_ID % 1000) + SLURM_ARRAY_TASK_ID))
PARTICIPANT=$(printf "P%03d" "${SLURM_ARRAY_TASK_ID}")

uv run python -m torch.distributed.run --nproc_per_node=4 --master_port="${PORT}" \
    main.py --config_file <config> --test_participants "${PARTICIPANT}"
```

Always derive the port from the job ID. Fixed ports collide across concurrent jobs.

For hyperparameter sweeps, use job arrays over config variations (or Hydra/W&B), on `gpu` for
quick iteration.

## Interactive Sessions

```bash
salloc --job-name=Interactive_Session --partition=pophealth \
    --nodes=1 --mem=160000 --ntasks=16 --gres=gpu:a100:1 --time=5:00:00

module load cuda
cd /mmfs1/data/group/pgh004/carrow/repo/remote-physiology
uv run python main.py --limit_windows 8 --test_participants P015 \
    --config_file configs/neckflix/NECKFLIX_PHYSMAMBA_SMOKE.yaml
exit    # release the allocation when done
```

## Monitoring

```bash
squeue -u $USER              # your jobs
watch -n 1 squeue -u $USER   # auto-refresh
sinfo -p medical             # partition availability
scontrol show job <jobid>    # why a job is pending
scancel <jobid>              # cancel
tail -f logs/<jobid>_*.out   # follow output
nvidia-smi                   # GPU status (compute session only)
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Job stuck pending | Partition busy | Follow the escalation ladder above |
| `CUDA not available` | Missing module | `module load cuda` in the script/session |
| Address already in use | Fixed master port | Derive `PORT` from `SLURM_JOB_ID` |
| Hangs at distributed init | `--nproc_per_node` ≠ `--gres` count | Make them match |
| OOM in `logs/*.err` | Batch/resolution too large | Lower batch size, image size, or chunk length in the YAML |

Different GPU types need different config tuning — image size, chunk length, and batch size in the
YAML often need adjusting when moving between V100, A100, and H100.

**Debugging workflow**: reproduce in `salloc` on a single GPU → shrink the config (lower resolution,
shorter chunks) → read `logs/*.err` → scale back up via `sbatch` once it works.

## Paths

- Working directory: `/mmfs1/data/group/pgh004/carrow/repo/remote-physiology` (also reachable as
  `/group/pgh004/carrow/repo/remote-physiology`, which is what the SLURM scripts use)
- Group storage: `/group/pgh004/` — accessible from compute nodes
- The cache preprocessor is a **git submodule**, and the HPC checkout predates
  it, so `dataset/cachers/neckflix` is an empty directory there until someone runs
  `git submodule update --init` once in the working directory (then
  `git -C dataset/cachers/neckflix checkout main`). An empty submodule does not fail
  loudly: `uv run --project dataset/cachers/neckflix` silently falls through to this
  project instead, downloads torch, and dies in a compiler. Check
  `test -f dataset/cachers/neckflix/pyproject.toml` before trusting a cache-build job.

## Building a Cache (CPU, no GPU)

Preprocessing is a SLURM job like any other — CPU partition, no `module load
cuda`, no `--gres`. Budget **~12 GB RAM per worker** (`--mem`); the event
camera dominates that, so `--perspectives 1 2` is both cheaper and avoids the
ECF HDF5 codec that only the docker image builds.

```bash
#SBATCH --partition=work
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "/group/pgh004/carrow/repo/remote-physiology"
uv run --project dataset/cachers/neckflix neckflix-preprocess \
    --input-dir <raw Neckflix root> --output-dir <cache dir> \
    --resize 256 256 --perspectives 1 2 --num-workers 2
uv run python tools/validate_cache.py <cache dir>
```

Model it on `.slurm_scripts/Neckflix_Unsupervised.slurm`, the other CPU-only
template. The submodule resolves from its own `uv.lock` and its own Python
3.12, so the first run in a fresh checkout syncs a second environment — do
that inside the job, not on the login node.

## Red Flags — Stop

- About to run `python train.py` / `uv run python main.py` straight from the shell
- "I'll just run this small thing on the login node"
- Writing a `.slurm` script from scratch instead of copying the template
- `--mem=0`, or a hardcoded `--master_port`
- `--nproc_per_node` that doesn't match the requested GPU count

**All of these mean: stop, and go through SLURM properly.**
