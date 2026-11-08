import pandas as pd


def fit_most_popular_all(train_ratings):

    """Rank items by total rating count across the training set."""

    return train_ratings["item_id"].value_counts()


def fit_most_popular_recent_positive(train_ratings, days=120):

    """Rank items by positive rating count (rating >= 4) within the
    most recent `days` of the training period.
    """

    ratings = train_ratings.copy()
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], utc=True)

    cutoff = ratings["timestamp"].max() - pd.Timedelta(days=days)

    recent_positive = ratings[
        (ratings["timestamp"] >= cutoff) &
        (ratings["rating"] >= 4)
    ]

    return recent_positive["item_id"].value_counts()


def recommend_most_popular(item_popularity, n=10):
    """Return the top-n most popular item IDs."""
    return item_popularity.head(n).index.tolist()
