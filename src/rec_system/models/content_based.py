"""Content-based profile matching on genres and release year"""

import numpy as np
import pandas as pd

from rec_system.metrics import DEFAULT_THRESHOLD, top_k_unseen

PROFILES = ("centred", "liked")


def item_features(
    movies: pd.DataFrame, year_bin: int = 10, year_weight: float = 1.0
) -> np.ndarray:
    """One row per movie: multi-hot genres, then a one-hot release-year bin (slide 4).

    The year part is multiplied by `year_weight`, so 0 means genres only.
    """
    genres = sorted(
        {genre for movie_genres in movies["genres"] for genre in movie_genres}
    )
    genre_part = np.array(
        [
            [genre in movie_genres for genre in genres]
            for movie_genres in movies["genres"]
        ],
        dtype=float,
    )

    year_bins = (movies["release_year"] // year_bin) * year_bin
    year_part = pd.get_dummies(year_bins).to_numpy(dtype=float) * year_weight

    return np.hstack([genre_part, year_part])


def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    """Divide each row by its length, so a dot product becomes a cosine."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def fit_content_based(
    user_item_matrix: pd.DataFrame,
    movies: pd.DataFrame,
    profile: str = "centred",
    year_bin: int = 10,
    year_weight: float = 1.0,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Build the item feature vectors. User profiles are made at recommend time.

    Every movie in `movies` can be recommended, including movies nobody rated
    in training: content-based only needs the item's description.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}")

    features = item_features(movies, year_bin, year_weight)
    return {
        "features": features,
        "unit_features": normalise_rows(features),
        "item_ids": movies["item_id"].to_numpy(),
        "row_of_item": {item: row for row, item in enumerate(movies["item_id"])},
        "profile": profile,
        "threshold": threshold,
    }


def user_profile(model_data: dict, user_ratings: pd.Series) -> np.ndarray:
    """Add up the feature vectors of the movies the user rated (slide 6).

    centred: weight each movie by rating - user's mean rating, so disliked
             movies push the profile away from their features.
    liked:   add only the movies rated >= threshold, each with weight 1.
    """
    rated = user_ratings[user_ratings > 0]
    rows = [model_data["row_of_item"][item] for item in rated.index]
    features = model_data["features"][rows]

    if model_data["profile"] == "centred":
        weights = rated.to_numpy(dtype=float) - rated.mean()
    else:
        weights = (rated.to_numpy() >= model_data["threshold"]).astype(float)
    return weights @ features


def recommend(
    model_data: dict, user_item_matrix: pd.DataFrame, user_id: int, n: int = 10
) -> list[int]:
    """Top-n unrated movies by cosine(profile, movie); empty for users with no ratings."""
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    profile = user_profile(model_data, user_ratings)
    if not profile.any():
        return []  # e.g. every rating equals the user's mean: no taste to match

    scores = model_data["unit_features"] @ normalise_rows(profile)
    seen = set(user_ratings.index[user_ratings.to_numpy() > 0])
    return top_k_unseen(scores, model_data["item_ids"], seen, k=n)