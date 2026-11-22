"""User-based kNN"""

import numpy as np
import pandas as pd


SIMILARITIES = ("cosine", "pearson", "jaccard")
AGGREGATIONS = ("sum", "average")
BLOCK_SIZE = 256

def user_similarity(user_item_matrix: pd.DataFrame, metric: str="cosine") -> np.ndarray:
    """user-by-user similarity matrix, with each user's similarity to itself set to 0"""
    ratings = user_item_matrix.to_numpy(dtype=np.float32)
    rated = ratings > 0

    if metric == "jaccard":
        binary = rated.astype(np.float32)
        overlap = binary @ binary.T
        counts = binary.sum(axis=1)
        union = counts[:, None] + counts[None, :] - overlap
        similarity = np.divide(
            overlap, union, out=np.zeros_like(overlap), where=union > 0
        )
    elif metric in ("cosine", "pearson"):
        if metric == "pearson":
            # Subtract each user's mean rating from the ratings they gave (slide 7)
            user_means = ratings.sum(axis=1, keepdims=True) / np.maximum(
                rated.sum(axis=1, keepdims=True), 1
            )
            ratings = np.where(rated, ratings - user_means, 0.0).astype(np.float32)
        norms = np.linalg.norm(ratings, axis=1)
        denominator = np.outer(norms, norms)
        similarity = np.divide(
            ratings @ ratings.T,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
    else:
        raise ValueError(f"metric must be one of {SIMILARITIES}, got {metric!r}")

    np.fill_diagonal(similarity, 0.0)
    return similarity


def fit_user_knn(
        user_item_matrix: pd.DataFrame,
        k: int=30,
        metric: str="cosine",
        aggregation: str="sum",
) -> dict:
    """precompute user similarities, neighbours are chosen per item at recommend time"""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
        )

    return {
        "similarity": user_similarity(user_item_matrix, metric),
        "ratings": user_item_matrix.to_numpy(dtype=np.float32),
        "user_rows": {user: row for row, user in enumerate(user_item_matrix.index)},
        "item_ids": user_item_matrix.columns.to_numpy(),
        "k": k,
        "metric": metric,
        "aggregation": aggregation,
    }


def score_items(model_data: dict, row: int) -> np.ndarray:
    """Score every item for the user in matrix row `row`.

    For each item, the neighbours are the k most similar users who rated it.
    Users with similarity <= 0 are never neighbours.
    Items the user rated, or that no neighbour rated, get -inf.
    """
    similarities = model_data["similarity"][row]
    ratings = model_data["ratings"]

    # Only positively similar users can be neighbours; most similar first
    candidates = np.flatnonzero(similarities > 0)
    candidates = candidates[np.argsort(-similarities[candidates], kind="stable")]

    n_items = ratings.shape[1]
    weighted = np.zeros(n_items)  # sum of sim(u, v) * r(v, i) over neighbours v
    total_similarity = np.zeros(n_items)  # sum of sim(u, v) over neighbours v
    raters_so_far = np.zeros(n_items, dtype=np.int16)

    # Walk down the users from most to least similar and, for each item, keep
    # the first k who rated it. Done in blocks of users only to save memory.
    for start in range(0, len(candidates), BLOCK_SIZE):
        block = candidates[start : start + BLOCK_SIZE]
        block_ratings = ratings[block]
        rated = block_ratings > 0

        rank_among_raters = np.cumsum(rated, axis=0, dtype=np.int16) + raters_so_far
        is_neighbour = rated & (rank_among_raters <= model_data["k"])
        raters_so_far = rank_among_raters[-1]

        block_sims = similarities[block]
        weighted += block_sims @ (block_ratings * is_neighbour)
        total_similarity += block_sims @ is_neighbour.astype(np.float32)

    scores = np.full(n_items, -np.inf)
    has_neighbour = total_similarity > 0
    if model_data["aggregation"] == "sum":
        scores[has_neighbour] = weighted[has_neighbour]
    else:
        scores[has_neighbour] = (
            weighted[has_neighbour] / total_similarity[has_neighbour]
        )

    scores[ratings[row] > 0] = -np.inf
    return scores

def recommend(
    model_data: dict, user_item_matrix: pd.DataFrame, user_id: int, n: int = 10
) -> list[int]:
    """Top-n unseen item IDs for a user; an empty list for users not seen in training."""
    row = model_data["user_rows"].get(user_id)
    if row is None:
        return []

    scores = score_items(model_data, row)
    ranked = np.argsort(-scores, kind="stable")
    ranked = ranked[np.isfinite(scores[ranked])][:n]
    return model_data["item_ids"][ranked].tolist()