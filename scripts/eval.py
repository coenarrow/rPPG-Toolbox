"""Score a run's test records — a stub, to be shaped once train and infer settle.

The third step of the train -> infer -> eval chain. Until this script is
written, evaluation runs through the package's own entry point:

    uv run python -m src.evaluation.evaluate RUN_DIR [RUN_DIR ...] [--out DIR]

which scores every absolute-class signal and every heart-rate source in
``RUN_DIR/test_records.pt`` and writes ``windows.csv``, ``rates.csv``,
``summary.csv``, ``digest.txt`` and the plots beside it (or under ``--out``,
required when pooling several runs).
"""

import sys


def main(argv=None) -> None:
    sys.exit("scripts/eval.py is a stub; evaluate with\n"
             "  uv run python -m src.evaluation.evaluate RUN_DIR")


if __name__ == "__main__":
    main()
