"""Score a run's records, one recording and camera at a time.

    uv run python scripts/eval.py runs/PHYSNET_PURE.01_202609091000
    uv run python scripts/eval.py runs/A/test_records runs/B/test_records

Each positional path is a records directory ``scripts/infer.py`` wrote, or a
run directory as shorthand for its ``test_records/``. No config, interface
file or checkpoint is needed: everything comes from the CSVs and
``meta.json``. Torch is still imported today, transitively through the
signal registry, so evaluating on a machine without it is not yet possible.
Every ``<recording>/<perspective>/`` folder is scored afresh
(``src/evaluation/recording.py``) and its ``beats.csv``, ``readings.csv``
and ``rates.csv`` written beside its trace tables. ``docs/evaluation.md``
lists every column.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.recording import score_recording                # noqa: E402
from src.outputs import META_NAME, RECORDS_DIR                      # noqa: E402

DEFAULT_READING_SECONDS = 30.0


def find_records_dirs(paths) -> list:
    """Each path as a records directory: itself when it holds ``meta.json``,
    its ``test_records/`` when it is a run directory."""
    dirs = []
    for path in map(Path, paths):
        if (path / META_NAME).is_file():
            dirs.append(path)
        elif (path / RECORDS_DIR / META_NAME).is_file():
            dirs.append(path / RECORDS_DIR)
        else:
            raise FileNotFoundError(
                f"{path} is neither a records directory (no {META_NAME}) nor a "
                f"run directory holding {RECORDS_DIR}/")
    return dirs


def recording_folders(records_dir: Path, meta: dict) -> list:
    """Every ``<recording>/<perspective>/`` folder under the records
    directory, found by the first trace's table."""
    first = str(meta["traces"][0])
    return sorted(path.parent for path in Path(records_dir).glob(f"*/*/{first}.csv"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score every recording of one or more records directories, "
                    "writing beats.csv, readings.csv and rates.csv beside its "
                    "trace tables.")
    parser.add_argument(
        "dirs", nargs="+", metavar="DIR",
        help="records directories (or run directories holding test_records/)")
    parser.add_argument(
        "--reading-seconds", type=float, default=DEFAULT_READING_SECONDS, metavar="S",
        help="length of one reading, the stretch every metric is scored over "
             "(default: 30)")
    return parser


def main(argv=None) -> list:
    """Score every recording; returns the folders scored, in order."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records_dirs = find_records_dirs(args.dirs)
    except FileNotFoundError as err:
        parser.error(str(err))
    scored = []
    for records_dir in records_dirs:
        meta = json.loads((records_dir / META_NAME).read_text(encoding="utf-8"))
        folders = recording_folders(records_dir, meta)
        if not folders:
            parser.error(f"no recordings found under {records_dir}")
        for folder in folders:
            score_recording(folder, meta, args.reading_seconds)
        scored.extend(folders)
        print(f"scored {len(folders)} recording(s) under {records_dir}")
    return scored


if __name__ == "__main__":
    main()
