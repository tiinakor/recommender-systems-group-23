"""Disaggregated evaluation across user and item groups.

Groups are defined on the final training data (train + validation), the data
the evaluated models were trained on:

- User activity: Low = at most the median number of ratings, High = above the
  80th percentile. Users in between belong to neither group.
- Item popularity: Popular = at least the 80th percentile of ratings per movie,
  Long-tail = everything else, including movies with no training ratings.

For user groups the metrics are averaged over the group's users. For item
groups the top-10 lists stay the same, but only test positives on that group's
movies count as relevant; users with no such positive are left out. Coverage is
the share of the catalogue (user groups) or of the group's movies (item groups)
recommended to at least one user. Seeded models are averaged over their seeds.
"""

import pandas as pd

from rec_system.metrics import DEFAULT_THRESHOLD, evaluate

METRICS = ["precision@10", "recall@10", "ndcg@10"]


# Users


def describe_user_activity(train):
    """Printout to identify quantiles for low- and high-activity users."""
    user_activity = train.groupby("user_id").size()

    print("\nUser activity statistics:")
    print(
        user_activity.describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
        .round(2)
        .to_frame()
        .T.to_string(index=False)
    )


def create_user_groups(train):
    """Identify low- and high-activity users, with a readable definition per group."""
    activity = train.groupby("user_id").size()

    low_threshold = activity.quantile(0.50)
    high_threshold = activity.quantile(0.80)

    groups = pd.Series(index=activity.index, dtype="object")
    groups[activity <= low_threshold] = "Low"
    groups[activity > high_threshold] = "High"

    definitions = {
        "Low": f"<= {low_threshold:.0f} training ratings (median)",
        "High": f"> {high_threshold:.0f} training ratings (80th percentile)",
    }
    return groups, definitions


# Items


def describe_item_interactions(train):
    interactions = train.groupby("item_id").size()

    print("\nItem interactions statistics:")
    print(
        interactions.describe(
            percentiles=[0.10, 0.20, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]
        )
    )

    print("\nInteraction count frequencies:")
    print(interactions.value_counts().sort_index().head(20))


def create_item_groups(train, catalogue):
    """Popular vs long-tail movies; movies nobody rated in training are long-tail."""
    interactions = train.groupby("item_id").size().reindex(catalogue, fill_value=0)
    popularity_threshold = train.groupby("item_id").size().quantile(0.80)

    groups = pd.Series("Long-tail", index=interactions.index, dtype="object")
    groups[interactions >= popularity_threshold] = "Popular"

    definitions = {
        "Popular": f">= {popularity_threshold:.0f} training ratings (80th percentile)",
        "Long-tail": f"< {popularity_threshold:.0f} training ratings",
    }
    return groups, definitions


# Evaluate one segment


def evaluate_segment(
    recommendations,
    held_out,
    n_catalogue_items,
    k=10,
    threshold=DEFAULT_THRESHOLD,
):
    """Evaluate recommendations for a particular test segment."""
    segment_recommendations = {
        user_id: recommendations.get(user_id, [])
        for user_id in held_out["user_id"].unique()
    }
    summary, _ = evaluate(
        recommendations=segment_recommendations,
        held_out=held_out,
        n_catalogue_items=n_catalogue_items,
        k=k,
        threshold=threshold,
    )
    return summary


# Multi-model disaggregated evaluation


def evaluate_all_models_disaggregated(
    recommendations_by_model,
    train,
    test,
    catalogue,
    n_catalogue_items,
    k=10,
    threshold=DEFAULT_THRESHOLD,
):
    """Evaluate any number of models on every user and item segment.

    `recommendations_by_model` maps a model name to {seed: {user_id: top-k list}}
    (deterministic models have a single seed, None). Returns one row per model,
    dimension and group, averaged over seeds.
    """
    user_groups, user_definitions = create_user_groups(train)
    item_groups, item_definitions = create_item_groups(train, catalogue)

    segments = []
    for group, definition in user_definitions.items():
        users = user_groups[user_groups == group].index
        segments.append(
            (
                "User activity",
                group,
                definition,
                test[test["user_id"].isin(users)],
                None,
            )
        )
    for group, definition in item_definitions.items():
        items = set(item_groups[item_groups == group].index)
        segments.append(
            (
                "Item popularity",
                group,
                definition,
                test[test["item_id"].isin(items)],
                items,
            )
        )

    rows = []
    for model, recommendations_by_seed in recommendations_by_model.items():
        for dimension, group, definition, segment_test, group_items in segments:
            per_seed = []
            for recommendations in recommendations_by_seed.values():
                summary = evaluate_segment(
                    recommendations, segment_test, n_catalogue_items, k, threshold
                )
                if group_items is not None:
                    recommended = {
                        i for recs in recommendations.values() for i in recs[:k]
                    }
                    summary["coverage"] = len(recommended & group_items) / len(
                        group_items
                    )
                per_seed.append(summary)

            seeds = [s for s in recommendations_by_seed if s is not None]
            rows.append(
                {
                    "model": model,
                    "dimension": dimension,
                    "group": group,
                    "definition": definition,
                    **{
                        m: sum(s[m] for s in per_seed) / len(per_seed)
                        for m in [*METRICS, "coverage"]
                    },
                    "n_users_evaluated": per_seed[0]["n_users_evaluated"],
                    "n_items_in_group": len(group_items) if group_items else None,
                    "seeds": ",".join(map(str, seeds))
                    if seeds
                    else "none (deterministic)",
                }
            )
    return pd.DataFrame(rows)
