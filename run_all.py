"""Reproduce every number and figure in the report."""

import argparse
from pathlib import Path

from rec_system.eda import run_eda
from rec_system.data import load_ratings
from rec_system.split import split_ratings

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Path to the ml-1m folder"
    )
    args = parser.parse_args()

    run_eda(args.data_dir, REPO_ROOT / "results", REPO_ROOT / "reports" / "figures")

    # Load and split data
    ratings = load_ratings(args.data_dir)
    train, validation, test = split_ratings(ratings)

if __name__ == "__main__":
    main()
