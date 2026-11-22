"""Small hand-checkable examples for user-based kNN and content-based."""

import numpy as np
import pandas as pd
import pytest

from rec_system.models import content_based, user_knn


@pytest.fixture
def matrix():
    # 0 = not rated. Ann and Bob agree on items 10 and 20; Cleo is the opposite.
    return pd.DataFrame(
        [[5, 4, 0, 0], [5, 5, 4, 1], [1, 0, 5, 5]],
        index=pd.Index([1, 2, 3], name="user_id"),  # Ann, Bob, Cleo
        columns=pd.Index([10, 20, 30, 40], name="item_id"),
    )


def test_cosine_similarity_between_users(matrix):
    sim = user_knn.user_similarity(matrix, "cosine")
    # Ann . Bob = 5*5 + 4*5 = 45, |Ann| = sqrt(41), |Bob| = sqrt(67)
    assert sim[0, 1] == pytest.approx(45 / np.sqrt(41 * 67), abs=1e-6)
    assert sim[0, 0] == 0.0  # a user is never their own neighbour


def test_user_knn_recommends_what_the_closest_user_liked(matrix):
    model = user_knn.fit_user_knn(matrix, k=1, metric="cosine", aggregation="sum")
    # k=1: for each item, only the most similar user who rated it counts.
    # For Ann, Bob (sim 0.86) beats Cleo (sim 0.11): Bob gave 30 a 4 and 40 a 1.
    assert user_knn.recommend(model, matrix, user_id=1, n=2) == [30, 40]


def test_user_knn_skips_rated_and_unknown_users(matrix):
    model = user_knn.fit_user_knn(matrix, k=2)
    assert 10 not in user_knn.recommend(model, matrix, user_id=1, n=4)
    assert user_knn.recommend(model, matrix, user_id=99) == []


@pytest.fixture
def movies():
    return pd.DataFrame(
        {
            "item_id": [10, 20, 30, 40, 50],
            "genres": [["Action"], ["Action", "Drama"], ["Comedy"], ["Drama"], ["Action"]],
            "release_year": [1995, 1996, 1995, 1980, 1999],
        }
    )


def test_item_features_genres_then_year_bins(movies):
    features = content_based.item_features(movies, year_bin=10, year_weight=0.5)
    # Columns: Action, Comedy, Drama | 1980s, 1990s
    assert features[0].tolist() == [1, 0, 0, 0, 0.5]
    assert features[3].tolist() == [0, 0, 1, 0.5, 0]


def test_centred_profile_and_cosine_match(matrix, movies):
    model = content_based.fit_content_based(
        matrix, movies, profile="centred", year_weight=0.0
    )
    # Bob: mean 3.75 -> weights +1.25, +1.25, +0.25, -2.75
    # Profile: Action 2.5, Comedy 0.25, Drama -1.5 (year columns are 0 here)
    profile = content_based.user_profile(model, matrix.loc[2])
    assert profile[:3].tolist() == pytest.approx([2.5, 0.25, -1.5])
    # Movie 50 (Action, never rated by anyone) is the only unseen movie left
    assert content_based.recommend(model, matrix, user_id=2, n=3) == [50]