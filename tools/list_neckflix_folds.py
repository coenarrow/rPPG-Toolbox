"""Print the LOSO folds available in a Neckflix zarr cache.

Construction is metadata-only, so this is cheap enough for a login node — it is
the intended way to fan a SLURM job array out over participants:

    uv run python tools/list_neckflix_folds.py --config_file <config> > folds.txt
    sbatch --array=1-$(wc -l < folds.txt) .slurm_scripts/Neckflix_PhysMamba_LOSO.slurm

``--attribute`` enumerates something other than ``participant`` (``posture``,
``session``, ``light``, ``perspective``), and ``--counts`` adds the number of
windows each value contributes.
"""

import argparse
import sys
from collections import Counter

sys.path.insert(0, ".")

from config import load_config                                   # noqa: E402
from dataset.data_loader.NeckflixLoader import NeckflixDataset   # noqa: E402
from dataset.data_loader.neckflix_config import zarr_config      # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--split", default="test",
                        choices=("train", "valid", "test", "unsupervised"))
    parser.add_argument("--attribute", default="participant")
    parser.add_argument("--counts", action="store_true",
                        help="also print the window count per value")
    parser.add_argument("--prefix", default="",
                        help="prepend to each value, e.g. --prefix P for P015")
    args = parser.parse_args()

    config = load_config(args.config_file)
    dataset = NeckflixDataset(zarr_config(config, args.split))

    if not args.counts:
        for value in dataset.attribute_values(args.attribute):
            print(f"{args.prefix}{value}")
        return

    counts = Counter()
    for recording, perspective, _start in dataset.windows:
        if args.attribute == "perspective":
            counts[str(perspective)] += 1
            continue
        attrs = dataset.dataset_dict[recording]["attrs"]
        if args.attribute in attrs:
            counts[str(attrs[args.attribute])] += 1
    for value in dataset.attribute_values(args.attribute):
        print(f"{args.prefix}{value}\t{counts[value]}")


if __name__ == "__main__":
    main()
