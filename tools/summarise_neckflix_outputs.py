"""Pool one run or a whole LOSO sweep into one clinical report.

``MultiSignalTrainer`` writes one self-describing record per scored window, so
everything here is recomputed offline without touching the zarr cache. Point it
at a single fold's pickle for that fold's numbers, or at a sweep directory to
reach the participant and cohort levels — a fold has exactly one test subject,
so no single run can produce a pooled band.

    uv run python tools/summarise_neckflix_outputs.py runs/neckflix_physmamba
    uv run python tools/summarise_neckflix_outputs.py <pickle> --csv windows.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

from evaluation.records import load                       # noqa: E402
from evaluation.report import build_frame, digest, write  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path,
                        help="an *_outputs.pickle, or a directory of them")
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write the tidy frame here")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="resamples for Pearson/CCC standard errors")
    parser.add_argument("--hr-method", default="FFT",
                        choices=("FFT", "Peak"))
    args = parser.parse_args()

    run = load(args.target)
    print(f"Pooled {len(run.windows)} windows at {run.fs} Hz "
          f"across signals {run.signals()}")
    frame = build_frame(run, bootstrap=args.bootstrap, hr_method=args.hr_method)
    summary = digest(frame, run)
    print(summary)
    # Always: the JSON is where the digest, the threshold provenance and the
    # unmet study-design notes live, and a CSV quoted without them is exactly
    # the artefact that turns an indicative grade into a claim. --csv adds a
    # copy at the caller's path, it does not replace the report.
    directory = args.target if args.target.is_dir() else args.target.parent
    write(frame, summary, directory, "pooled")
    if args.csv:
        frame.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
