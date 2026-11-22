"""Disaggregated evaluation across user and item groups."""

import pandas as pd
from rec_system.metrics import evaluate


METRICS = ["precision@10", "recall@10", "ndcg@10"]


# Users

def describe_user_activity(train):
    """Printout to identify quantiles for low- and high-activity users."""
    user_activity = train.groupby("user_id").size()

    print("\nUser activity statistics:")
    print(
        user_activity.describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50,
                         0.75, 0.90, 0.95, 0.99]
        ).round(2).to_frame().T.to_string(index=False)
    )


def create_user_groups(train):
    """Identify low- and high-activity users."""

    activity = train.groupby("user_id").size()

    low_threshold = activity.quantile(0.50)
    high_threshold = activity.quantile(0.80)

    groups = pd.Series(index=activity.index, dtype="object")

    groups[activity <= low_threshold] = "Low"
    groups[activity > high_threshold] = "High"

    return groups


## Items

def describe_item_interactions(train):
    interactions = train.groupby("item_id").size()

    print("\nItem interactions statistics:")
    print(
        interactions.describe(
            percentiles=[0.10, 0.20, 0.25, 0.50,
                        0.75, 0.80, 0.90, 0.95, 0.99]
        )
    )

    print("\nInteraction count frequencies:")
    print(interactions.value_counts().sort_index().head(20))




def create_item_groups(train):
    """Identify item interaction and popularity groups."""

    interactions = train.groupby("item_id").size()

    # Few vs Many
    low_interaction_threshold = interactions.quantile(0.40)
    high_interaction_threshold = interactions.quantile(0.80)

    interaction_groups = pd.Series(index=interactions.index, dtype="object")
    interaction_groups[ interactions <= low_interaction_threshold] = "Few"
    interaction_groups[interactions >= high_interaction_threshold] = "Many"

    # Popular vs Long-tail: 80th percentile
    popularity_threshold = interactions.quantile(0.80)

    popularity_groups = pd.Series(
        index=interactions.index,
        dtype="object",
    )

    popularity_groups[
        interactions >= popularity_threshold
    ] = "Popular"

    popularity_groups[
        interactions < popularity_threshold
    ] = "Long-tail"

    return interaction_groups, popularity_groups



# --------------------------------------------------
# EVALUATE ONE SEGMENT
# --------------------------------------------------

def evaluate_segment(
    recommendations,
    held_out,
    n_catalogue_items,
    k=10,
    threshold=4,
    debug=False,
):
    """Evaluate recommendations for a particular test segment."""

    segment_users = held_out["user_id"].unique()

    segment_recommendations = {
        user_id: recommendations.get(user_id, [])
        for user_id in segment_users
    }

    summary, per_user = evaluate(
        recommendations=segment_recommendations,
        held_out=held_out,
        n_catalogue_items=n_catalogue_items,
        k=k,
        threshold=threshold,
    )

    if debug:
        relevant = held_out[held_out["rating"] >= threshold]

        recommended_items = {
            item
            for recs in segment_recommendations.values()
            for item in recs[:k]
        }

        relevant_items_set = set(relevant["item_id"])

        print("\n--- SEGMENT DEBUG ---")
        print("Test interactions:", len(held_out))
        print("Positive interactions:", len(relevant))
        print("Users evaluated:", len(per_user))
        print("Unique relevant items:", len(relevant_items_set))
        print("Unique recommended items:", len(recommended_items))
        print(
            "Relevant items appearing in recommendations:",
            len(relevant_items_set & recommended_items),
        )

    return {
        metric: summary[metric]
        for metric in METRICS
    }


# --------------------------------------------------
# MULTI-MODEL DISAGGREGATED EVALUATION
# --------------------------------------------------

def evaluate_all_models_disaggregated(
    model_results,
    train,
    test,
    n_catalogue_items,
    k=10,
    threshold=4,
):
    """
    Evaluate any number of recommendation models.

    Expected input:

    {
        "Model 1": {
            "recommendations": model_recommendations,
        },
        "Model 2": {
            "recommendations": model_recommendations,
        },
    }
    """

    user_groups = create_user_groups(train)
    interaction_groups, popularity_groups = create_item_groups(train)

    # Define segments once for all models
    segments = [
        ("Overall", "All", None),
        ("User activity", "Low", user_groups),
        ("User activity", "High", user_groups),
        ("Item interactions", "Few", interaction_groups),
        ("Item interactions", "Many", interaction_groups),
        ("Item popularity", "Popular", popularity_groups),
        ("Item popularity", "Long-tail", popularity_groups),
    ]

    results = []

    for model_name, model_data in model_results.items():

        recommendations = model_data["recommendations"]

        for dimension, segment, groups in segments:

            if dimension == "Overall":
                segment_test = test

            elif dimension == "User activity":
                segment_users = groups[groups == segment].index

                segment_test = test[
                    test["user_id"].isin(segment_users)
                ]

            else:
                segment_items = groups[groups == segment].index

                segment_test = test[
                    test["item_id"].isin(segment_items)
                ]

            if segment_test.empty:
                continue

            metrics = evaluate_segment(
                recommendations=recommendations,
                held_out=segment_test,
                n_catalogue_items=n_catalogue_items,
                k=k,
                threshold=threshold,
            )

            for metric, value in metrics.items():
                results.append({
                    "evaluation_dimension": dimension,
                    "segment": segment,
                    "metric": metric,
                    "model": model_name,
                    "value": value,
                })

    return pd.DataFrame(results)


# --------------------------------------------------
# presentation table
# --------------------------------------------------

def create_presentation_table(results):

    table = results.pivot(
        index=["evaluation_dimension", "segment"],
        columns=["model", "metric"],
        values="value",
    )

    table.columns = [
        f"{model}_{metric}"
        for model, metric in table.columns
    ]

    return table.reset_index()