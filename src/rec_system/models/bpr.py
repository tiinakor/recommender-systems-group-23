"""Bayesian Personalized Ranking"""

import numpy as np
import pandas as pd

from rec_system.metrics import DEFAULT_THRESHOLD, top_k_unseen


def _prepare_data(
    user_item_matrix: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the rating matrix into BPR training pairs.

    Returns the row (user) and column (item) of every liked cell, and a
    users x items True/False table of what each user liked.
    """
    liked = user_item_matrix.to_numpy() >= threshold
    users, items = np.nonzero(liked)
    return users, items, liked


def _sample_negatives(
    users: np.ndarray, liked: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """For each user, draw a random item they did not like (uniform, slide 12)."""
    negatives = rng.integers(liked.shape[1], size=len(users))
    clash = liked[users, negatives]
    while clash.any():
        negatives[clash] = rng.integers(liked.shape[1], size=clash.sum())
        clash = liked[users, negatives]
    return negatives


def _sgd_step(
    P: np.ndarray,
    Q: np.ndarray,
    b: np.ndarray,
    u: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    learning_rate: float,
    reg: float,
) -> float:
    """One SGD update on a batch of (u, i, j) triples; returns the mean ln sigma(x)."""
    p_u, q_i, q_j = P[u], Q[i], Q[j]

    # x = score(u, i) - score(u, j), with score = p_u . q + item bias
    x = np.sum(p_u * (q_i - q_j), axis=1) + b[i] - b[j]

    # d ln sigma(x) / dx = sigma(-x): large when the pair is ranked wrongly
    g = 1.0 / (1.0 + np.exp(x))
    g_col = g[:, None]

    # Gradient ascent on ln sigma(x) - reg * ||theta||^2
    np.add.at(P, u, learning_rate * (g_col * (q_i - q_j) - reg * p_u))
    np.add.at(Q, i, learning_rate * (g_col * p_u - reg * q_i))
    np.add.at(Q, j, learning_rate * (-g_col * p_u - reg * q_j))
    np.add.at(b, i, learning_rate * (g - reg * b[i]))
    np.add.at(b, j, learning_rate * (-g - reg * b[j]))

    # ln sigma(x), written so it does not overflow for large |x|
    return float(-np.mean(np.logaddexp(0.0, -x)))


def fit_bpr(
    user_item_matrix: pd.DataFrame,
    k: int = 64,
    learning_rate: float = 0.05,
    reg: float = 0.05,
    epochs: int = 20,
    batch_size: int = 1024,
    seed: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Learn user and item factors by SGD on sampled (user, liked, not-liked) triples."""
    rng = np.random.default_rng(seed)
    users, items, liked = _prepare_data(user_item_matrix, threshold)

    n_users, n_items = liked.shape
    P = rng.normal(0.0, 0.1, size=(n_users, k))
    Q = rng.normal(0.0, 0.1, size=(n_items, k))
    b = np.zeros(n_items)

    log_likelihood = []
    for _ in range(epochs):
        order = rng.permutation(len(users))
        batch_values = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            u, i = users[batch], items[batch]
            j = _sample_negatives(u, liked, rng)
            batch_values.append(_sgd_step(P, Q, b, u, i, j, learning_rate, reg))
        log_likelihood.append(float(np.mean(batch_values)))

    return {
        "P": P,
        "Q": Q,
        "b": b,
        "user_rows": {user: row for row, user in enumerate(user_item_matrix.index)},
        "item_ids": user_item_matrix.columns.to_numpy(),
        "log_likelihood": log_likelihood,
    }


def recommend(
    model_data: dict, user_item_matrix: pd.DataFrame, user_id: int, n: int = 10
) -> list[int]:
    """Top-n unseen items by BPR score; an empty list for users not seen in training."""
    row = model_data["user_rows"].get(user_id)
    if row is None:
        return []
    scores = model_data["Q"] @ model_data["P"][row] + model_data["b"]
    rated = user_item_matrix.loc[user_id]
    seen = set(rated.index[rated.to_numpy() > 0])
    return top_k_unseen(scores, model_data["item_ids"], seen, k=n)
