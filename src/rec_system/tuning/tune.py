"""Tune every model with the same budget on validation, then measure seed variability.

Lecture 3.1, slide 10: tune on validation only, give every model the same budget,
report variability over several seeds, and compare models with a paired test
across users.
"""

import random
from itertools import combinations, product

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


def fit_and_recommend(
    spec: ModelSpec, params: dict, train_matrix: pd.DataFrame, users, k: int = 10
) -> dict[int, list[int]]:
    """Fit one model on `train_matrix` and return a top-k list for each user."""
    model_data = spec.fit(train_matrix, **params)
    return {
        user_id: spec.recommend(model_data, train_matrix, user_id, n=k)
        for user_id in users
    }


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
    recommendations = fit_and_recommend(
        spec, params, train_matrix, relevant_items(held_out, threshold), k
    )
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
    metric: str = "ndcg@10",
    k: int = 10,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Refit with each seed and evaluate; deterministic models are run once.

    Returns one summary row per seed, each user's `metric` averaged over the
    seeds (indexed by user_id) for the paired test, and the top-k lists per seed
    (used for the disaggregated evaluation).
    """
    users = relevant_items(held_out, DEFAULT_THRESHOLD)
    rows, per_user_runs, recommendations_by_seed = [], [], {}
    for seed in seeds if spec.uses_seed else (None,):
        fit_params = {**params, "seed": seed} if spec.uses_seed else params
        recommendations = fit_and_recommend(spec, fit_params, train_matrix, users, k)
        summary, per_user = evaluate(
            recommendations=recommendations,
            held_out=held_out,
            n_catalogue_items=n_catalogue_items,
            k=k,
        )
        rows.append(
            {"model": spec.name, "seed": seed, **{m: summary[m] for m in METRICS}}
        )
        per_user_runs.append(per_user.set_index("user_id")[metric])
        recommendations_by_seed[seed] = recommendations

    per_user_mean = pd.concat(per_user_runs, axis=1).mean(axis=1).rename(spec.name)
    return pd.DataFrame(rows), per_user_mean, recommendations_by_seed


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


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted.tolist()


def paired_tests(
    per_user: dict[str, pd.Series], metric: str = "ndcg@10"
) -> pd.DataFrame:
    """Wilcoxon signed-rank test on per-user scores for every pair of models."""
    rows = []
    for a, b in combinations(per_user, 2):
        pair = pd.concat([per_user[a], per_user[b]], axis=1, join="inner")
        diff = pair[a] - pair[b]
        result = stats.wilcoxon(pair[a], pair[b])
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                f"{metric}_a": pair[a].mean(),
                f"{metric}_b": pair[b].mean(),
                "mean_difference": diff.mean(),
                "a_better": int((diff > 0).sum()),
                "b_better": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "n_users": len(pair),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    table = pd.DataFrame(rows)
    table["p_value_holm"] = holm(table["p_value"].tolist())
    return table
