"""Load the MovieLens 1M data"""

from pathlib import Path

import pandas as pd


def load_ratings(data_dir: str | Path) -> pd.DataFrame:
    ratings = pd.read_csv(
        Path(data_dir) / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return ratings
