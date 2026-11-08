import pandas as pd


def split_ratings(ratings):

    "Chronologically split each user's ratings into training, validation, and test sets (approximately 70/15/15)"

    splits = {"train": [], "validation": [], "test": []}

    for _, group in ratings.groupby("user_id"):
        group = group.sort_values("timestamp")
        n = len(group)

        n_test = round(n * 0.15)
        n_val = round(n * 0.15)

        train_end = n - n_val - n_test
        val_end = n - n_test

        splits["train"].append(group.iloc[:train_end])
        splits["validation"].append(group.iloc[train_end:val_end])
        splits["test"].append(group.iloc[val_end:])

    return tuple(
        pd.concat(splits[k]).reset_index(drop=True)
        for k in ("train", "validation", "test")
    )
