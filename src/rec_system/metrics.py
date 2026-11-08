"""Evaluation metrics for top-k recommendation

Implements Precision@k, Recall@k, NDCG@k and catalogue coverage. All metrics use k = 10

Conventions:
    1. Candidate set: the full catalogue. Each user's training items are removed before taking the top k. No neg sampling
    2. Relevance: a held-out item is relevant if its rating >= 4. Other items, including ones that never rated, is not relevant
    3. Users: only users with at least 1 relevant held-out item are scored. A scored user with no rec list gets 0, not skipped
    4. Averaging: metrics are computed per user, then averaged with equal weight per user

A recommendation list is a sequence of items id, best first; relevant and seen items are sets of item ids
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_K = 10
DEFAULT_THRESHOLD = 4


# --- building the inputs ---
def relevant_items(held_out: pd.DataFrame, threshold: float=DEFAULT_THRESHOLD) -> dict[int, set[int]]:
    """map each user to the set of held-out items they rated >= threshold
    users whose held-out ratings are all below the threshold do not appear."""
    positives = held_out[held_out["rating"] >= threshold]
    return {
        int(user): set(map(int, items)) for user, items in positives.groupby("user_id")["item_id"]
    }


def seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    """map each user to the items they interacted with in training"""
    return {
        int(user): set(map(int, items)) for user, items in train.groupby("user_id")["item_id"]
    }


def top_k_unseen(scores: np.ndarray, item_ids: Sequence[int], seen: Collection[int]=(), k: int=DEFAULT_K) -> list[int]:
    """return the ids of k highest-scoring items the user has not seen"""
    scores = np.asarray(scores, dtype=float).copy()         # copy -> dont change the model's array
    item_ids = np.asarray(item_ids)
    if seen:
        scores[np.isin(item_ids, list(seen))] = -np.inf
    k = min(k, int(np.isfinite(scores).sum()))              # cant returnn more that what's left
    if k <= 0:
        return []
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in item_ids[order]]


# --- per-list metrics ---
def precision_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """hits in the top k / k
    of the k movies we showed, how many did the user like?"""
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """hits in the top k / number of relevant items
    of all movies the user liked, how many did we show?"""
    if not relevant:
        return float("nan")
    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """each hit at rank r is worth: 1 / log2(r+1)
    a hit near the top is worth more than one near the bottom"""
    if not relevant:
        return float("nan")
    dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    n_ideal = min(k, len(relevant))             # how many hits a perfect list can have
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, n_ideal + 1))
    return dcg / idcg


# --- catalogue-level metrics ---
def catalogue_coverage(
        recommendations: Mapping[int, Sequence[int]],
        n_catalogue_items: int,
        k: int = DEFAULT_K
) -> float:
    """fraction of the catalogue appearing in top-k list
    computed over the lists of every user who received recommendations, not only the users who have relevant test items"""
    recommended: set[int] = set()
    for ranked in recommendations.values():
        recommended.update(ranked[:k])
    return len(recommended) / n_catalogue_items


# --- all together ---
def per_user_metrics(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, Collection[int]],
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """one row per evaluated user with precision, recall and NDCG at k
    a user is evaluated when they have at least one relevant held-out item.
    a user with relevant items but no recommendation list scores zero.
    """
    rows = []
    for user, rel in relevant.items():
        if not rel:
            continue
        ranked = list(recommendations.get(user, []))
        rows.append(
            {
                "user_id": user,
                "n_relevant": len(rel),
                f"precision@{k}": precision_at_k(ranked, rel, k),
                f"recall@{k}": recall_at_k(ranked, rel, k),
                f"ndcg@{k}": ndcg_at_k(ranked, rel, k),
            }
        )
    columns = ["user_id", "n_relevant", f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]
    return pd.DataFrame(rows, columns=columns).sort_values("user_id", ignore_index=True)


def evaluate(
    recommendations: Mapping[int, Sequence[int]],
    held_out: pd.DataFrame,
    n_catalogue_items: int,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[dict[str, float], pd.DataFrame]:
    """score a set of top-k lists against a held-out split.
 
    returns ``(summary, per_user)``. ``summary`` holds the mean of each
    per-user metric, catalogue coverage, and the user counts needed to report
    the protocol honestly.
    """
    relevant = relevant_items(held_out, threshold)
    per_user = per_user_metrics(recommendations, relevant, k)
    summary = {
        f"precision@{k}": float(per_user[f"precision@{k}"].mean()),
        f"recall@{k}": float(per_user[f"recall@{k}"].mean()),
        f"ndcg@{k}": float(per_user[f"ndcg@{k}"].mean()),
        "coverage": catalogue_coverage(recommendations, n_catalogue_items, k),
        "n_users_evaluated": len(per_user),
        "n_users_in_held_out": int(held_out["user_id"].nunique()),
        "n_users_without_recommendations": int(
            sum(1 for u in relevant if not recommendations.get(u))
        ),
    }
    return summary, per_user
 