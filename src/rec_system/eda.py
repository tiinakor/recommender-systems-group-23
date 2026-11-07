"""Describe the MovieLens 1M data and save the Part 1 figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from rec_system.data import load_movies, load_ratings, load_users

SERIES_COLOR = "#2a78d6"
GRID_COLOR = "#e5e4e0"


def summarize(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> dict:
    """Return the headline numbers used to describe the dataset."""
    per_item = ratings["item_id"].value_counts()
    per_user = ratings["user_id"].value_counts()
    n_top_items = int(len(per_item) * 0.1)
    user_times = ratings.groupby("user_id")["timestamp"]
    active_days = (user_times.max() - user_times.min()).dt.total_seconds() / 86400

    return {
        "n_ratings": len(ratings),
        "n_users": len(users),
        "n_movies_in_catalogue": len(movies),
        "n_movies_rated": len(per_item),
        "density_rated_movies": len(ratings) / (len(per_user) * len(per_item)),
        "ratings_per_user_median": float(per_user.median()),
        "ratings_per_user_min": int(per_user.min()),
        "ratings_per_user_max": int(per_user.max()),
        "ratings_per_item_median": float(per_item.median()),
        "ratings_per_item_max": int(per_item.max()),
        "items_with_under_10_ratings": int((per_item < 10).sum()),
        "ratings_share_of_top_10pct_items": float(
            per_item.iloc[:n_top_items].sum() / len(ratings)
        ),
        "rating_share": ratings["rating"]
        .value_counts(normalize=True)
        .sort_index()
        .to_dict(),
        "ratings_share_by_year": (
            ratings["timestamp"]
            .dt.year.value_counts(normalize=True)
            .sort_index()
            .to_dict()
        ),
        "user_active_days_median": float(active_days.median()),
        "share_of_users_active_within_1_day": float((active_days <= 1).mean()),
    }


def share_of_test_users_seen_in_train(
    ratings: pd.DataFrame, train_fraction: float
) -> dict:
    """Cut all ratings at one date and check which later users have earlier history."""
    ordered = ratings.sort_values("timestamp")
    cut_index = int(len(ordered) * train_fraction)
    train = ordered.iloc[:cut_index]
    test = ordered.iloc[cut_index:]
    test_users = test["user_id"].unique()
    seen = pd.Series(test_users).isin(train["user_id"])
    return {
        "train_fraction": train_fraction,
        "cutoff": str(test["timestamp"].iloc[0]),
        "n_test_users": len(test_users),
        "share_of_test_users_seen_in_train": float(seen.mean()),
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_long_tail(ratings: pd.DataFrame, path: Path) -> None:
    """Plot ratings per item and per user, most active first, on a log scale."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    panels = [("item_id", "Movies", "Movie rank"), ("user_id", "Users", "User rank")]
    for ax, (column, title, x_label) in zip(axes, panels):
        counts = ratings[column].value_counts().to_numpy()
        ax.plot(range(1, len(counts) + 1), counts, color=SERIES_COLOR, linewidth=2)
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"{x_label} (most ratings first)")
        _style_axis(ax)
    axes[0].set_ylabel("Number of ratings (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rating_distribution(ratings: pd.DataFrame, path: Path) -> None:
    """Plot the share of ratings at each star value."""
    shares = ratings["rating"].value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(shares.index, shares.to_numpy(), color=SERIES_COLOR, width=0.7)
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Share of all ratings (%)")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_monthly_activity(ratings: pd.DataFrame, path: Path) -> None:
    """Plot the number of ratings entered each month."""
    monthly = ratings.set_index("timestamp").resample("MS").size()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(monthly.index, monthly.to_numpy(), color=SERIES_COLOR, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Month the rating was entered")
    ax.set_ylabel("Number of ratings (log scale)")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_eda(data_dir: str | Path, results_dir: Path, figures_dir: Path) -> dict:
    """Compute the dataset summary."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users = load_users(data_dir)

    summary = summarize(ratings, movies, users)
    summary["global_cut_check"] = share_of_test_users_seen_in_train(
        ratings, train_fraction=0.8
    )
    (results_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))

    plot_long_tail(ratings, figures_dir / "long_tail.png")
    plot_rating_distribution(ratings, figures_dir / "rating_distribution.png")
    plot_monthly_activity(ratings, figures_dir / "monthly_activity.png")
    return summary
