"""Sparsity ablation: retrain on 50% and 20% of the training interactions."""

import pandas as pd

from rec_system.matrices import build_user_item_matrix
from rec_system.metrics import DEFAULT_THRESHOLD, evaluate, relevant_items, seen_items
from rec_system.models.registry import ModelSpec
from rec_system.tuning.tune import METRICS, SEEDS, mean_ci

FRACTIONS = (0.5, 0.2)


def recommend_unseen(
    spec: ModelSpec,
    model_data: dict,
    matrix: pd.DataFrame,
    user_id: int,
    seen_full: set,
    k: int = 10,
) -> list[int]:
    """Top-k from the model after removing items rated in the full training data."""
    seen_in_subset = (
        set(matrix.columns[matrix.loc[user_id].to_numpy() > 0])
        if user_id in matrix.index
        else set()
    )
    extra = len(seen_full - seen_in_subset)
    ranked = spec.recommend(model_data, matrix, user_id, n=k + extra)
    return [item for item in ranked if item not in seen_full][:k]


def sparsity_runs(
    spec: ModelSpec,
    params: dict,
    train_ratings: pd.DataFrame,
    test: pd.DataFrame,
    n_catalogue_items: int,
    fractions: tuple[float, ...] = FRACTIONS,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """One row per (fraction, seed): the model refitted on a random share of `train_ratings`."""
    seen_full = seen_items(train_ratings)
    rows = []
    for fraction in fractions:
        for seed in seeds:
            subset = train_ratings.sample(frac=fraction, random_state=seed)
            matrix = build_user_item_matrix(subset)
            fit_params = {**params, "seed": seed} if spec.uses_seed else params
            model_data = spec.fit(matrix, **fit_params)
            recommendations = {
                user_id: recommend_unseen(
                    spec, model_data, matrix, user_id, seen_full.get(user_id, set())
                )
                for user_id in relevant_items(test, DEFAULT_THRESHOLD)
            }
            summary, _ = evaluate(
                recommendations=recommendations,
                held_out=test,
                n_catalogue_items=n_catalogue_items,
            )
            rows.append(
                {
                    "model": spec.name,
                    "training_fraction": fraction,
                    "seed": seed,
                    **{m: summary[m] for m in METRICS},
                }
            )
            print(
                f"[sparsity {spec.name} {fraction:.0%} seed {seed}] "
                f"ndcg@10 {summary['ndcg@10']:.4f}"
            )
    return pd.DataFrame(rows)


def summarise_sparsity(runs: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    """Mean and 95% CI per model and fraction, plus each model's rank within a fraction."""
    rows = []
    for (model, fraction), group in runs.groupby(["model", "training_fraction"]):
        mean, low, high = mean_ci(group[metric])
        rows.append(
            {
                "model": model,
                "training_fraction": fraction,
                f"{metric}_mean": mean,
                f"{metric}_ci_low": low,
                f"{metric}_ci_high": high,
                "coverage_mean": group["coverage"].mean(),
                "n_seeds": len(group),
            }
        )
    summary = pd.DataFrame(rows)
    summary["rank"] = (
        summary.groupby("training_fraction")[f"{metric}_mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return summary.sort_values(
        ["training_fraction", "rank"], ascending=[False, True], ignore_index=True
    )
