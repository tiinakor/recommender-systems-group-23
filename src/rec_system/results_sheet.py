"""Write results.csv with the exact sheet names and columns of the course Results Sheet."""

from pathlib import Path

import pandas as pd

from rec_system.models.baseline.most_popular import VARIANTS

PART1_SHEET = "Part 1 - Baselines"
PART1_COLUMNS = [
    "Model",
    "Variant / settings",
    "Similarity",
    "k",
    "Precision@10",
    "Recall@10",
    "NDCG@10",
    "Catalogue coverage",
    "Seed",
    "Notes",
]
NO_SEED = "none (deterministic)"


def describe_most_popular(positive_only: bool, window_days: int | None) -> str:
    """Readable settings for one Most Popular variant."""
    counted = "positive ratings only (>= 4)" if positive_only else "all ratings"
    period = f"last {window_days} days" if window_days else "all time"
    return f"{counted}, {period}"


def part1_rows(
    test_summaries: list[dict], best_params: dict, n_catalogue_items: int
) -> pd.DataFrame:
    """One row per Part 1 baseline, from the summaries evaluate_model returned on test."""
    rows = []
    for summary in test_summaries:
        model = summary["model"]
        if model == "item_knn":
            row = {
                "Model": "Item-based kNN",
                "Variant / settings": (
                    f"aggregation={best_params['aggregation']}; best of 30 configs "
                    "by validation NDCG@10; refit on train + validation"
                ),
                "Similarity": best_params["metric"],
                "k": best_params["k"],
            }
        else:
            variant = model.removeprefix("most_popular_")
            row = {
                "Model": "Most Popular",
                "Variant / settings": describe_most_popular(*VARIANTS[variant]),
                "Similarity": "n/a",
                "k": "n/a",
            }
        row.update(
            {
                "Precision@10": round(summary["precision@10"], 4),
                "Recall@10": round(summary["recall@10"], 4),
                "NDCG@10": round(summary["ndcg@10"], 4),
                "Catalogue coverage": round(summary["coverage"], 4),
                "Seed": NO_SEED,
                "Notes": (
                    f"test set, {summary['n_users_evaluated']} users; "
                    f"catalogue of {n_catalogue_items} rated movies"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=PART1_COLUMNS)


def write_results_csv(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """Stack every sheet's rows into one CSV, with the sheet name as the first column."""
    frames = [rows.assign(Sheet=name) for name, rows in sheets.items()]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["Sheet", *[c for c in combined.columns if c != "Sheet"]]]
    combined.to_csv(path, index=False)
