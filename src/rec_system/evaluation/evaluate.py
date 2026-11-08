
import pandas as pd

from rec_system.matrices import build_user_item_matrix
from rec_system.metrics import evaluate, relevant_items

def evaluate_model(
    model_name,
    model_data,
    recommend_fn,
    train_matrix,
    test,
    n_catalogue_items,
    k=10,
    threshold=4,
):
    """Generate recommendations and evaluate a model."""

    relevant = relevant_items(test, threshold)

    recommendations = {}

    for user_id in relevant:
        recommendations[user_id] = recommend_fn(
            model_data,
            train_matrix,
            user_id,
            n=k,
        )

    # Calculate evaluation metrics using metrics.py
    summary, per_user = evaluate(
        recommendations=recommendations,
        held_out=test,
        n_catalogue_items=n_catalogue_items,
        k=k,
        threshold=threshold,
    )

    summary["model"] = model_name

    return summary, per_user, recommendations


def fit_and_evaluate_model(
    model_name,
    fit_fn,
    recommend_fn,
    train,
    validation,
    test,
    n_catalogue_items,
    model_params=None,
    k=10,
    threshold=4,
):
    """Fit final model and evaluate on test data."""

    if model_params is None:
        model_params = {}

    # Combine training and validation
    final_train = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    # Build final user-item matrix
    final_matrix = build_user_item_matrix(final_train)

    # Fit model
    model_data = fit_fn(
        final_matrix,
        **model_params,
    )

    # Evaluate
    summary, per_user, recommendations = evaluate_model(
        model_name=model_name,
        model_data=model_data,
        recommend_fn=recommend_fn,
        train_matrix=final_matrix,
        test=test,
        n_catalogue_items=n_catalogue_items,
        k=k,
        threshold=threshold,
    )

    return summary, per_user, recommendations


def save_evaluation_results(
    summary,
    per_user,
    results_dir,
    model_name,
):
    """Save evaluation summary and per-user metrics."""

    results_dir.mkdir(parents=True, exist_ok=True)

    filename = model_name.lower().replace(" ", "_")

    pd.DataFrame([summary]).to_csv(
        results_dir / f"{filename}_evaluation.csv",
        index=False,
    )

    per_user.to_csv(
        results_dir / f"{filename}_per_user.csv",
        index=False,
    )
