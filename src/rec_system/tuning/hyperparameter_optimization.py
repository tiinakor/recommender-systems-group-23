import time
import pandas as pd

from rec_system.models.baseline.item_knn import fit_item_knn, recommend
from rec_system.metrics import evaluate, relevant_items


def tune_item_knn(
    train_matrix: pd.DataFrame,
    validation_data: pd.DataFrame,
    n: int = 10,
    threshold: float = 4,
    n_catalogue_items: int | None = None,
):
    """
    Tune Item-KNN hyperparameters using validation Recall@K.

    Uses the shared evaluate() function to calculate:
    Precision@K, Recall@K, NDCG@K and catalogue coverage.
    """

    k_values = [2, 3, 4, 6, 10]
    distance_metrics = ["cosine", "euclidean"]

    results = []

    start_time = time.time()

    configurations = [
        (metric, k)
        for metric in distance_metrics
        for k in k_values
    ]

    if n_catalogue_items is None:
        n_catalogue_items = train_matrix.shape[1]

    validation_users = list(
        relevant_items(validation_data, threshold).keys()
    )

    print("\nStarting Item-KNN hyperparameter tuning...")

    for i, (metric, k) in enumerate(configurations, start=1):

        config_start = time.time()

        print(
            f"\n[{i}/{len(configurations)}] "
            f"Training Item-KNN | k={k}, metric={metric}"
        )

        # Fit model
        model = fit_item_knn(
            train_matrix,
            k=k,
            metric=metric,
        )

        # Generate recommendations for validation users
        recommendations = {}

        for user_id in validation_users:
            recommendations[int(user_id)] = recommend(
                model,
                train_matrix,
                user_id,
                n=n,
            )

        # Evaluate using shared metrics
        summary, _ = evaluate(
            recommendations=recommendations,
            held_out=validation_data,
            n_catalogue_items=n_catalogue_items,
            k=n,
            threshold=threshold,
        )

        score = summary[f"recall@{n}"]

        results.append({
            "k": k,
            "metric": metric,
            f"precision@{n}": summary[f"precision@{n}"],
            f"recall@{n}": score,
            f"ndcg@{n}": summary[f"ndcg@{n}"],
            "coverage": summary["coverage"],
        })

        elapsed = time.time() - start_time
        remaining = elapsed / i * (len(configurations) - i)

        print(f"Recall@{n}: {score:.4f}")

        print(
            f"Configuration time: "
            f"{time.time() - config_start:.2f} seconds"
        )

        print(
            f"Estimated remaining: "
            f"{remaining / 60:.2f} minutes"
        )

    # Sort configurations by validation Recall@K
    results_df = (
        pd.DataFrame(results)
        .sort_values(
            f"recall@{n}",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Select best configuration
    best = results_df.iloc[0]

    best_params = {
        "k": int(best["k"]),
        "metric": str(best["metric"]),
    }

    print("\nItem-KNN tuning completed.")
    print(f"Best parameters: {best_params}")

    print(
        f"Best validation Recall@{n}: "
        f"{best[f'recall@{n}']:.4f}"
    )

    return results_df, best_params