"""Implicit-feedback ALS"""

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from rec_system.metrics import DEFAULT_THRESHOLD, top_k_unseen


def fit_als(
    user_item_matrix: pd.DataFrame,
    factors: int = 16,
    regularization: float = 50,
    alpha: float = 1.0,
    iterations: int = 15,
    seed: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Fit ALS on 1/0 liked data, with confidence 1 + alpha for liked cells (slide 10)."""
    liked = (user_item_matrix.to_numpy() >= threshold).astype(np.float32)

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
    )
    model.fit(csr_matrix(liked), show_progress=False)

    return {
        "P": np.asarray(model.user_factors),
        "Q": np.asarray(model.item_factors),
        "user_rows": {user: row for row, user in enumerate(user_item_matrix.index)},
        "item_ids": user_item_matrix.columns.to_numpy(),
    }


def recommend(
    model_data: dict, user_item_matrix: pd.DataFrame, user_id: int, n: int = 10
) -> list[int]:
    """Top-n unrated items by ALS score; an empty list for users not seen in training."""
    row = model_data["user_rows"].get(user_id)
    if row is None:
        return []

    scores = model_data["Q"] @ model_data["P"][row]
    rated = user_item_matrix.loc[user_id]
    seen = set(rated.index[rated.to_numpy() > 0])
    return top_k_unseen(scores, model_data["item_ids"], seen, k=n)
