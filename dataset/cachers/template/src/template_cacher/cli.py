"""<dataset>-preprocess: align + resize one dataset's recordings into zarr stores.

Dataset-agnostic apart from the two constants at the top. The flow per
recording is: skip if a complete store already covers the request; create
the store with its root attrs; read timestamps and traces per stream and
align them; decode, resize and write each requested modality; stamp
complete. Recordings run in parallel and fail independently.
"""
import argparse
import multiprocessing
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import zarr
from tqdm import tqdm

import template_cacher
from template_cacher.scan import RecordingInfo, discover_recordings, scan_recording
from template_cacher.traces import (
    TRACE_UNITS, AlignedStream, align_stream, read_stream_traces,
)
from template_cacher.video import decode_video
from template_cacher.writer import (
    NOMINAL_FPS,
    covers,
    init_store,
    mark_complete,
    write_perspective_group,
    write_trace,
    write_video_group,
)

#: FILL IN. The modalities this dataset can write (contract vocabulary:
#: gr, rgb, ir, depth, t) and its perspective names. Both become the
#: ``--modalities`` / ``--perspectives`` choices and defaults.
MODALITIES: tuple[str, ...] = ("rgb",)
PERSPECTIVES: tuple[str, ...] = ("1",)


@dataclass
class Job:
    """One recording's work order (picklable for multiprocessing)."""
    recording: RecordingInfo
    output_dir: Path
    resize: tuple[int, int] | None
    modalities: list[str]
    perspectives: list[str]
    overwrite: bool


@dataclass
class RecordingResult:
    name: str
    status: str  # "cached" | "skipped" | "failed"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    detail: str | None = None  # short content summary for the live status line


def _align_streams(job: Job, result: RecordingResult) -> dict[tuple[str, str], AlignedStream]:
    """Read timestamps and traces for every stream this run writes, and align them."""
    rec = job.recording
    aligned: dict[tuple[str, str], AlignedStream] = {}
    for perspective in sorted(rec.perspectives):
        if perspective not in job.perspectives:
            continue
        for modality, info in rec.perspectives[perspective].items():
            if modality not in job.modalities:
                continue
            timestamps, traces = read_stream_traces(info)
            unknown = set(traces) - set(TRACE_UNITS)
            if unknown:
                raise ValueError(
                    f"{perspective}/{modality}: traces {sorted(unknown)} are not in "
                    f"TRACE_UNITS; every trace needs a declared unit"
                )
            stream = align_stream(info.num_frames, timestamps, traces)
            for name, count in stream.interior_nans().items():
                result.warnings.append(
                    f"{perspective}/{modality}: {count} interior NaN sample(s) in "
                    f"{name}; kept as-is"
                )
            aligned[(perspective, modality)] = stream
    return aligned


def _write_video(
    root: zarr.Group,
    job: Job,
    aligned: dict[tuple[str, str], AlignedStream],
    result: RecordingResult,
) -> tuple[int, int]:
    """Decode, resize and write every aligned stream. Returns (modalities, frames) written."""
    rec = job.recording
    groups: dict[str, zarr.Group] = {}
    modalities_written = frames_written = 0
    for (perspective, modality), stream in aligned.items():
        info = rec.perspectives[perspective][modality]
        frames, fps = decode_video(info, modality, num_frames=stream.num_frames, resize=job.resize)
        if round(fps) != round(NOMINAL_FPS):
            result.warnings.append(
                f"{perspective}/{modality}: container rate {fps:.4f} is not "
                f"{NOMINAL_FPS:g}; keeping nominal fps"
            )
        decoded = frames.shape[1]
        if decoded < stream.num_frames:
            result.warnings.append(
                f"{perspective}/{modality}: decoded {decoded} frames < "
                f"{stream.num_frames} from metadata; truncating timestamps/traces"
            )
            stream = stream.truncated(decoded)
        if perspective not in groups:
            groups[perspective] = write_perspective_group(root, perspective, fps=NOMINAL_FPS)
        modality_group = groups[perspective].create_group(modality)
        write_video_group(modality_group, frames, timestamps_us=stream.timestamps_us)
        for name, values in stream.traces.items():
            write_trace(modality_group, name, values, units=TRACE_UNITS[name])
        modalities_written += 1
        frames_written += stream.num_frames
    return modalities_written, frames_written


def process_recording(job: Job) -> RecordingResult:
    """Align, decode/resize and write one recording's zarr store."""
    rec = job.recording
    store_path = job.output_dir / f"{rec.name}.zarr"
    result = RecordingResult(name=rec.name, status="cached")
    try:
        if not job.overwrite and covers(store_path, job.modalities, job.perspectives, job.resize):
            result.status = "skipped"
            return result
        root = init_store(
            store_path,
            {
                **rec.attrs,
                "source_resolution": rec.source_resolution,
                "resized_to": list(job.resize) if job.resize else None,
                "tool_version": template_cacher.__version__,
            },
        )
        aligned = _align_streams(job, result)
        modalities_written, frames_written = _write_video(root, job, aligned, result)
        result.detail = f"{modalities_written} modalities, {frames_written} frames"
        mark_complete(root, job.modalities, job.perspectives)
    except Exception:
        result.status = "failed"
        result.error = traceback.format_exc()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="template-preprocess",
        description="Temporally align (and optionally resize) recordings into zarr stores.",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Dataset root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("H", "W"), default=None,
                        help="Target frame size (default: native)")
    parser.add_argument("--recordings", nargs="+", default=None,
                        help="Recording names to process (default: all)")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES),
                        choices=list(MODALITIES), help="Modalities to write (default: all)")
    parser.add_argument("--perspectives", nargs="+", default=list(PERSPECTIVES),
                        choices=list(PERSPECTIVES), help="Perspectives to write (default: all)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Recordings processed in parallel (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild stores that are already marked complete")
    parser.add_argument("--version", action="version", version=template_cacher.__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        available = discover_recordings(args.input_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1
    names = args.recordings if args.recordings is not None else available
    if not names:
        print("No recordings found.")
        return 1

    # Probe each recording separately so a corrupt file in one cannot abort
    # discovery of the rest.
    print(f"Found {len(names)} recording{'s' if len(names) != 1 else ''}; scanning metadata...",
          flush=True)
    jobs: list[Job] = []
    scan_failures: list[RecordingResult] = []
    for name in tqdm(names, desc="scanning", unit="rec", leave=False):
        try:
            rec = scan_recording(args.input_dir, name)
        except Exception:
            scan_failures.append(
                RecordingResult(name=name, status="failed", error=traceback.format_exc()))
            continue
        jobs.append(Job(recording=rec, output_dir=args.output_dir,
                        resize=tuple(args.resize) if args.resize else None,
                        modalities=args.modalities, perspectives=args.perspectives,
                        overwrite=args.overwrite))

    if not jobs and not scan_failures:
        print("No recordings found.")
        return 1
    for r in scan_failures:
        tqdm.write(f"FAILED (scan) {r.name}")

    def report(result: RecordingResult) -> None:
        line = f"{result.status} {result.name}"
        if result.detail:
            line += f" ({result.detail})"
        tqdm.write(line)
        for w in result.warnings:
            tqdm.write(f"WARNING [{result.name}]: {w}")

    workers = max(args.num_workers, 1)
    print(f"Processing {len(jobs)} recording{'s' if len(jobs) != 1 else ''} "
          f"with {workers} worker{'s' if workers > 1 else ''}...", flush=True)
    results: list[RecordingResult] = list(scan_failures)
    if workers <= 1:
        for job in tqdm(jobs, desc="recordings"):
            result = process_recording(job)
            report(result)
            results.append(result)
    else:
        with multiprocessing.Pool(workers) as pool:
            for result in tqdm(pool.imap_unordered(process_recording, jobs),
                               total=len(jobs), desc="recordings"):
                report(result)
                results.append(result)

    cached = [r for r in results if r.status == "cached"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    for r in failed:
        print(f"FAILED [{r.name}]:\n{r.error}")
    print(f"{len(cached)} cached, {len(skipped)} skipped, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
