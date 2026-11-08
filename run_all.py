"""Reproduce every number and figure in the report."""

import argparse
from pathlib import Path

from rec_system.eda import run_eda
from rec_system.data import load_ratings
from rec_system.split import split_ratings
from rec_system.matrices import build_user_item_matrix
from rec_system.tuning.hyperparameter_optimization import tune_item_knn

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
    n_catalogue_items = ratings["item_id"].nunique()
    
    train, validation, test = split_ratings(ratings)

    train_matrix = build_user_item_matrix(train)

    tuning_results, best_params = tune_item_knn(
        train_matrix=train_matrix,
        validation_data=validation,
        n=10,
        threshold=4,
        n_catalogue_items=n_catalogue_items,
    )


if __name__ == "__main__":
    main()
