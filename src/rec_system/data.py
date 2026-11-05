"""Load the MovieLens 1M data"""

from pathlib import Path

import pandas as pd


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    """Load ratings.dat"""
    ratings = pd.read_csv(
        Path(data_dir) / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings


def load_movies(data_dir: str | Path) -> pd.DataFrame:
    """Load movies.dat, extracting release year and genres and new columns"""
    movies = pd.read_csv(
        Path(data_dir) / "movies.dat",
        sep="::",
        engine="python",
        header=None,
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["release_year"] = (
        movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype(int)
    )
    movies["genres"] = movies["genres"].str.split("|")
    return movies


def load_users(data_dir: str | Path) -> pd.DataFrame:
    """Load users.dat"""
    return pd.read_csv(
        Path(data_dir) / "users.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"zip_code": str},
    )
