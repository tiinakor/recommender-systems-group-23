import pandas as pd

from rec_system.metrics import DEFAULT_THRESHOLD, top_k_unseen

# (positive_only, window_days): every combination of the two choices in the brief
VARIANTS = {
    "all_ratings_all_time": (False, None),
    "positive_all_time": (True, None),
    "all_ratings_recent": (False, 120),
    "positive_recent": (True, 120),
}


def fit_most_popular(
    ratings: pd.DataFrame,
    positive_only: bool = False,
    window_days: int | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Count ratings per item, optionally only positive ones and only the last `window_days`.

    Takes the ratings table rather than the user-item matrix because the window needs timestamps.
    """
    counted = ratings
    if positive_only:
        counted = counted[counted["rating"] >= threshold]
    if window_days is not None:
        cutoff = ratings["timestamp"].max() - pd.Timedelta(days=window_days)
        counted = counted[counted["timestamp"] >= cutoff]

    popularity = counted["item_id"].value_counts()
    return {
        "item_ids": popularity.index.to_numpy(),
        "scores": popularity.to_numpy(dtype=float),
        "positive_only": positive_only,
        "window_days": window_days,
    }


def recommend(
    model_data: dict, user_item_matrix: pd.DataFrame, user_id: int, n: int = 10
) -> list[int]:
    """Top-n most popular items the user has not rated; users with no ratings get the plain top-n."""
    seen = ()
    if user_id in user_item_matrix.index:
        row = user_item_matrix.loc[user_id]
        seen = set(row.index[row.to_numpy() > 0])
    return top_k_unseen(model_data["scores"], model_data["item_ids"], seen, k=n)
