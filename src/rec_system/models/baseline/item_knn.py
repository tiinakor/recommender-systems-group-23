import numpy as np
import pandas as pd

SIMILARITIES = ("cosine", "adjusted_cosine", "jaccard")
AGGREGATIONS = ("sum", "average")


def item_similarity(
    user_item_matrix: pd.DataFrame, metric: str = "cosine"
) -> np.ndarray:
    """Item-by-item similarity matrix, with each item's similarity to itself set to 0."""
    ratings = user_item_matrix.to_numpy(dtype=float)
    rated = ratings > 0

    if metric == "jaccard":
        binary = rated.astype(float)
        overlap = binary.T @ binary
        counts = binary.sum(axis=0)
        union = counts[:, None] + counts[None, :] - overlap
        similarity = np.divide(
            overlap, union, out=np.zeros_like(overlap), where=union > 0
        )
    elif metric in ("cosine", "adjusted_cosine"):
        if metric == "adjusted_cosine":
            # Subtract each user's mean rating from the ratings they gave (slide 6)
            user_means = ratings.sum(axis=1, keepdims=True) / np.maximum(
                rated.sum(axis=1, keepdims=True), 1
            )
            ratings = np.where(rated, ratings - user_means, 0.0)
        norms = np.linalg.norm(ratings, axis=0)
        denominator = np.outer(norms, norms)
        similarity = np.divide(
            ratings.T @ ratings,
            denominator,
            out=np.zeros((len(norms), len(norms))),
            where=denominator > 0,
        )
    else:
        raise ValueError(f"metric must be one of {SIMILARITIES}, got {metric!r}")

    np.fill_diagonal(similarity, 0.0)
    return similarity


def fit_item_knn(
    user_item_matrix: pd.DataFrame,
    k: int = 30,
    metric: str = "cosine",
    aggregation: str = "sum",
) -> dict:
    """Precompute item similarities; neighbours are chosen per user at recommend time."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
        )

    return {
        "similarity": item_similarity(user_item_matrix, metric),
        "item_ids": user_item_matrix.columns.to_numpy(),
        "k": k,
        "metric": metric,
        "aggregation": aggregation,
    }


def score_items(model_data: dict, user_ratings: np.ndarray) -> np.ndarray:
    """Score every item for one user from their k most similar rated items.

    Rated items and items with no positively similar rated neighbour get -inf.
    """
    rated = np.flatnonzero(user_ratings > 0)
    scores = np.full(len(user_ratings), -np.inf)
    if len(rated) == 0:
        return scores

    # Similarity of every candidate item to each item the user rated
    similarities = model_data["similarity"][:, rated]
    neighbour_ratings = np.broadcast_to(user_ratings[rated], similarities.shape)

    # Keep the k rated items most similar to each candidate (slide 5, step 2)
    k = model_data["k"]
    if len(rated) > k:
        top = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        similarities = np.take_along_axis(similarities, top, axis=1)
        neighbour_ratings = np.take_along_axis(neighbour_ratings, top, axis=1)

    # Negative neighbours are dropped (slide 10)
    similarities = np.clip(similarities, 0.0, None)
    weighted = (similarities * neighbour_ratings).sum(axis=1)
    total_similarity = similarities.sum(axis=1)

    has_neighbour = total_similarity > 0
    if model_data["aggregation"] == "sum":
        scores[has_neighbour] = weighted[has_neighbour]
    else:
        scores[has_neighbour] = (
            weighted[has_neighbour] / total_similarity[has_neighbour]
        )

    scores[rated] = -np.inf
    return scores


def recommend(
    model_data: dict, user_item_matrix: pd.DataFrame, user_id: int, n: int = 10
) -> list[int]:
    """Top-n unseen item IDs for a user; an empty list for users with no ratings."""
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = (
        user_item_matrix.loc[user_id]
        .reindex(model_data["item_ids"], fill_value=0)
        .to_numpy(dtype=float)
    )
    scores = score_items(model_data, user_ratings)

    ranked = np.argsort(-scores, kind="stable")
    ranked = ranked[np.isfinite(scores[ranked])][:n]
    return model_data["item_ids"][ranked].tolist()
