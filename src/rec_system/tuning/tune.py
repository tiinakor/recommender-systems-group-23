"""Tune every model with the same budget on validation, then measure seed variability."""

import random
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from rec_system.metrics import DEFAULT_THRESHOLD, evaluate, relevant_items
from rec_system.models.registry import ModelSpec

BUDGET = 30  # configurations per model; item-kNN's full grid has exactly 30
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("precision@10", "recall@10", "ndcg@10", "coverage", "n_users_evaluated")


def sample_configs(
    search_space: dict, budget: int = BUDGET, seed: int = 0
) -> list[dict]:
    """The full grid if it fits the budget, otherwise `budget` configs drawn at random."""
    grid = [
        dict(zip(search_space, values)) for values in product(*search_space.values())
    ]
    if len(grid) <= budget:
        return grid
    return random.Random(seed).sample(grid, budget)


def score_model(
    spec: ModelSpec,
    params: dict,
    train_matrix: pd.DataFrame,
    held_out: pd.DataFrame,
    n_catalogue_items: int,
    k: int = 10,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[dict, pd.DataFrame]:
    """Fit one model on `train_matrix` and evaluate it on `held_out`."""
    model_data = spec.fit(train_matrix, **params)
    recommendations = {
        user_id: spec.recommend(model_data, train_matrix, user_id, n=k)
        for user_id in relevant_items(held_out, threshold)
    }
    return evaluate(
        recommendations=recommendations,
        held_out=held_out,
        n_catalogue_items=n_catalogue_items,
        k=k,
        threshold=threshold,
    )


def tune_model(
    spec: ModelSpec,
    train_matrix: pd.DataFrame,
    validation: pd.DataFrame,
    n_catalogue_items: int,
    budget: int = BUDGET,
    select_by: str = "ndcg@10",
) -> tuple[pd.DataFrame, dict]:
    """Score `budget` configurations on validation users with training history; pick the best."""
    known_validation = validation[validation["user_id"].isin(train_matrix.index)]
    configs = sample_configs(spec.search_space, budget)

    rows = []
    for i, params in enumerate(configs, start=1):
        fit_params = {**params, "seed": 0} if spec.uses_seed else params
        summary, _ = score_model(
            spec, fit_params, train_matrix, known_validation, n_catalogue_items
        )
        rows.append({"model": spec.name, **params, **{m: summary[m] for m in METRICS}})
        print(
            f"[{spec.name} {i}/{len(configs)}] {params} -> {select_by} {summary[select_by]:.4f}"
        )

    # max() keeps the first config on ties, so the choice is reproducible
    best_index = max(range(len(rows)), key=lambda i: rows[i][select_by])
    results = (
        pd.DataFrame(rows)
        .sort_values(select_by, ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    return results, configs[best_index]


def seed_runs(
    spec: ModelSpec,
    params: dict,
    train_matrix: pd.DataFrame,
    held_out: pd.DataFrame,
    n_catalogue_items: int,
    seeds: tuple[int, ...] = SEEDS,
) -> pd.DataFrame:
    """Refit with each seed and evaluate; deterministic models are run once."""
    rows = []
    for seed in seeds if spec.uses_seed else (None,):
        fit_params = {**params, "seed": seed} if spec.uses_seed else params
        summary, _ = score_model(
            spec, fit_params, train_matrix, held_out, n_catalogue_items
        )
        rows.append(
            {"model": spec.name, "seed": seed, **{m: summary[m] for m in METRICS}}
        )
    return pd.DataFrame(rows)


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-based confidence interval over seeds."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * stats.sem(values)
    return mean, float(mean - half), float(mean + half)


def summarise_seeds(runs: pd.DataFrame, metric: str = "ndcg@10") -> dict:
    """One row per model: mean, 95% CI and number of seeds for `metric`."""
    mean, low, high = mean_ci(runs[metric])
    return {
        "model": runs["model"].iloc[0],
        f"{metric}_mean": mean,
        f"{metric}_ci_low": low,
        f"{metric}_ci_high": high,
        "n_seeds": len(runs),
    }
