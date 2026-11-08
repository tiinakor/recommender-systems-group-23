"""Reproduce every number and figure in the report."""

import argparse
import json
from pathlib import Path

import pandas as pd

from rec_system.data import load_ratings
from rec_system.eda import run_eda
from rec_system.evaluation.evaluate import evaluate_model, save_evaluation_results
from rec_system.matrices import build_user_item_matrix
from rec_system.metrics import relevant_items
from rec_system.metrics_check import check_against_sklearn, check_slide_8
from rec_system.models.baseline import item_knn, most_popular
from rec_system.split import split_ratings
from rec_system.tuning.hyperparameter_optimization import tune_item_knn

REPO_ROOT = Path(__file__).resolve().parent
SUMMARY_COLUMNS = [
    "model",
    "precision@10",
    "recall@10",
    "ndcg@10",
    "coverage",
    "n_users_evaluated",
]


def evaluate_most_popular_variants(
    ratings: pd.DataFrame,
    matrix: pd.DataFrame,
    held_out: pd.DataFrame,
    n_catalogue_items: int,
) -> list[dict]:
    """Fit every Most Popular variant on `ratings` and score it on `held_out`."""
    summaries = []
    for name, (positive_only, window_days) in most_popular.VARIANTS.items():
        model = most_popular.fit_most_popular(
            ratings, positive_only=positive_only, window_days=window_days
        )
        summary, _, _ = evaluate_model(
            f"most_popular_{name}",
            model,
            most_popular.recommend,
            matrix,
            held_out,
            n_catalogue_items,
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Path to the ml-1m folder"
    )
    args = parser.parse_args()

    results_dir = REPO_ROOT / "results"
    run_eda(args.data_dir, results_dir, REPO_ROOT / "reports" / "figures")

    # Load and split data
    ratings = load_ratings(args.data_dir)
    n_catalogue_items = ratings["item_id"].nunique()

    train, validation, test = split_ratings(ratings)

    train_matrix = build_user_item_matrix(train)

    # Tune Item-KNN on validation users who have training history
    tuning_results, best_params = tune_item_knn(
        train_matrix=train_matrix,
        validation_data=validation,
        n=10,
        threshold=4,
        n_catalogue_items=n_catalogue_items,
    )
    tuning_results.to_csv(results_dir / "item_knn_tuning.csv", index=False)
    (results_dir / "item_knn_best_params.json").write_text(
        json.dumps(best_params, indent=2)
    )

    # Most Popular variants on the same validation users, for a fair comparison
    known_validation = validation[validation["user_id"].isin(train_matrix.index)]
    validation_summaries = evaluate_most_popular_variants(
        train, train_matrix, known_validation, n_catalogue_items
    )
    pd.DataFrame(validation_summaries)[SUMMARY_COLUMNS].to_csv(
        results_dir / "most_popular_validation.csv", index=False
    )

    # Refit every baseline on train + validation and score it once on test
    final_train = pd.concat([train, validation], ignore_index=True)
    final_matrix = build_user_item_matrix(final_train)

    knn_model = item_knn.fit_item_knn(final_matrix, **best_params)
    knn_summary, knn_per_user, knn_recommendations = evaluate_model(
        "item_knn",
        knn_model,
        item_knn.recommend,
        final_matrix,
        test,
        n_catalogue_items,
    )
    save_evaluation_results(knn_summary, knn_per_user, results_dir, "item_knn_test")

    test_summaries = [knn_summary] + evaluate_most_popular_variants(
        final_train, final_matrix, test, n_catalogue_items
    )
    test_results = pd.DataFrame(test_summaries)[SUMMARY_COLUMNS]
    test_results.to_csv(results_dir / "part1_test_results.csv", index=False)

    print("\nPart 1 test results:")
    print(test_results.round(4).to_string(index=False))

    # The brief requires self-written metrics to agree with a reference on a sample
    slide_check = check_slide_8()
    slide_check.to_csv(results_dir / "metrics_check_slide8.csv", index=False)
    sklearn_check = check_against_sklearn(knn_recommendations, relevant_items(test))
    sklearn_check.to_csv(results_dir / "metrics_check_sklearn.csv", index=False)
    max_gap = (sklearn_check["ndcg@10"] - sklearn_check["ndcg@10_sklearn"]).abs().max()

    print("\nMetrics check against Lecture 3.1, slide 8:")
    print(slide_check.round(2).to_string(index=False))
    print(
        f"Metrics check against scikit-learn on {len(sklearn_check)} test users: "
        f"largest NDCG@10 difference {max_gap:.2e}"
    )


if __name__ == "__main__":
    main()
