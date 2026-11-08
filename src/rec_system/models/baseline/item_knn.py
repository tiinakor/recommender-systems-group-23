import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_baseline_item_knn(user_item_matrix, k=30, metric="cosine"):
    """Fit Item-KNN and precompute item neighbours."""

    item_user_matrix = user_item_matrix.T.astype(float)

    n_neighbors = min(k + 1, len(item_user_matrix))

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="brute",
        n_jobs=-1
    )

    model.fit(item_user_matrix)

    distances, indices = model.kneighbors(item_user_matrix)

    return {
        "model": model,
        "distances": distances,
        "indices": indices,
        "item_user_matrix": item_user_matrix,
        "k": k,
        "metric": metric
    }


def recommend(model_data, user_item_matrix, user_id, n=10):
    """
    Generate top-n recommendations using Item-KNN.

    Excludes training items and returns item IDs.
    """

    item_user_matrix = model_data["item_user_matrix"]

    if user_id not in item_user_matrix.columns:
        return []

    ratings = item_user_matrix[user_id].to_numpy(dtype=float)

    rated = np.flatnonzero(ratings > 0)

    if len(rated) == 0:
        return []

    scores = np.zeros(len(item_user_matrix))
    weights = np.zeros(len(item_user_matrix))

    for item in rated:

        neighbors = model_data["indices"][item]
        distances = model_data["distances"][item]

        mask = neighbors != item

        neighbors = neighbors[mask][:model_data["k"]]
        distances = distances[mask][:model_data["k"]]

        if model_data["metric"] == "cosine":
            similarities = 1 - distances
        else:
            similarities = 1 / (1 + distances)

        mask = similarities > 0

        neighbors = neighbors[mask]
        similarities = similarities[mask]

        np.add.at(
            scores,
            neighbors,
            similarities * ratings[item]
        )

        np.add.at(
            weights,
            neighbors,
            similarities
        )

    predictions = np.divide(
        scores,
        weights,
        out=np.zeros_like(scores),
        where=weights > 0
    )

    # Exclude items already rated in training
    predictions[rated] = -np.inf

    # Exclude items without a prediction
    predictions[weights == 0] = -np.inf

    # Rank candidate items
    ranked = np.argsort(-predictions, kind="stable")

    # Return item IDs
    recommended_indices = ranked[
        np.isfinite(predictions[ranked])
    ][:n]

    return item_user_matrix.index[
        recommended_indices
    ].tolist()
