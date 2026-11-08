"""Tune Item-KNN on the validation set only (Lecture 3.1, slide 10)."""

import time
from itertools import product

import pandas as pd

from rec_system.metrics import evaluate, relevant_items
from rec_system.models.baseline.item_knn import (
    AGGREGATIONS,
    SIMILARITIES,
    fit_item_knn,
    recommend,
)

K_VALUES = (10, 20, 50, 100, 200)
SELECTION_METRICS = ("ndcg", "recall", "precision")


def tune_item_knn(
    train_matrix: pd.DataFrame,
    validation_data: pd.DataFrame,
    n: int = 10,
    threshold: float = 4,
    n_catalogue_items: int | None = None,
    select_by: str = "ndcg",
) -> tuple[pd.DataFrame, dict]:
    """
    Score every Item-KNN configuration on validation and pick the best by `select_by`@n.

    Only validation users who also appear in training are scored: Item-KNN cannot
    recommend to a user with no ratings, so the others would add a zero to every
    configuration.
    """
    if select_by not in SELECTION_METRICS:
        raise ValueError(f"select_by must be one of {SELECTION_METRICS}")

    if n_catalogue_items is None:
        n_catalogue_items = train_matrix.shape[1]

    known_validation = validation_data[
        validation_data["user_id"].isin(train_matrix.index)
    ]
    validation_users = list(relevant_items(known_validation, threshold))
    n_excluded = len(relevant_items(validation_data, threshold)) - len(validation_users)

    configurations = list(product(SIMILARITIES, K_VALUES, AGGREGATIONS))
    print(
        f"\nStarting Item-KNN tuning: {len(configurations)} configurations, "
        f"{len(validation_users)} validation users "
        f"({n_excluded} without training history excluded)"
    )

    results = []
    start_time = time.time()

    for i, (metric, k, aggregation) in enumerate(configurations, start=1):
        model = fit_item_knn(train_matrix, k=k, metric=metric, aggregation=aggregation)

        recommendations = {
            int(user_id): recommend(model, train_matrix, user_id, n=n)
            for user_id in validation_users
        }

        summary, _ = evaluate(
            recommendations=recommendations,
            held_out=known_validation,
            n_catalogue_items=n_catalogue_items,
            k=n,
            threshold=threshold,
        )

        results.append(
            {
                "metric": metric,
                "k": k,
                "aggregation": aggregation,
                f"precision@{n}": summary[f"precision@{n}"],
                f"recall@{n}": summary[f"recall@{n}"],
                f"ndcg@{n}": summary[f"ndcg@{n}"],
                "coverage": summary["coverage"],
                "n_users_evaluated": summary["n_users_evaluated"],
                "n_users_excluded_without_history": n_excluded,
            }
        )

        elapsed = time.time() - start_time
        remaining = elapsed / i * (len(configurations) - i)
        print(
            f"[{i}/{len(configurations)}] metric={metric}, k={k}, "
            f"aggregation={aggregation} | {select_by}@{n}: "
            f"{summary[f'{select_by}@{n}']:.4f} | "
            f"~{remaining / 60:.1f} min left"
        )

    # Stable sort keeps the grid order when two configurations tie
    score_column = f"{select_by}@{n}"
    results_df = (
        pd.DataFrame(results)
        .sort_values(score_column, ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    best = results_df.iloc[0]
    best_params = {
        "k": int(best["k"]),
        "metric": str(best["metric"]),
        "aggregation": str(best["aggregation"]),
    }

    print(f"\nBest parameters: {best_params}")
    print(f"Best validation {score_column}: {best[score_column]:.4f}")

    return results_df, best_params
