
import pandas as pd


def split_ratings(ratings):
    """Split ratings globally into chronological 70/15/15 sets."""

    ratings = ratings.sort_values("timestamp").reset_index(drop=True)

    # Cut by the number of ratings, not by calendar dates
    n = len(ratings)
    train_cutoff = ratings.iloc[int(n * 0.70) - 1]["timestamp"]
    val_cutoff = ratings.iloc[int(n * 0.85) - 1]["timestamp"]

    train = ratings[ratings["timestamp"] <= train_cutoff].copy()
    validation = ratings[
        (ratings["timestamp"] > train_cutoff) &
        (ratings["timestamp"] <= val_cutoff)
    ].copy()
    test = ratings[ratings["timestamp"] > val_cutoff].copy()

    # Remove users without training history
    train_users = set(train["user_id"].unique())

    val_new_users = set(validation["user_id"].unique()) - train_users
    test_new_users = set(test["user_id"].unique()) - train_users

    validation = validation[validation["user_id"].isin(train_users)].copy()
    test = test[test["user_id"].isin(train_users)].copy()

    # Print split statistics
    print("\nGlobal chronological split")
    print("-" * 40)
    print(f"Total ratings:       {n}")
    print(f"Training cutoff:     {train_cutoff}")
    print(f"Validation cutoff:   {val_cutoff}")

    print("\nUsers without training history:")
    print(f"Validation: {len(val_new_users)}")
    print(f"Test:       {len(test_new_users)}")

    print("\nSplit statistics:")
    for name, split in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test)
    ]:
        print(f"{name:<12} {len(split):>8} rows, "
              f"{split['user_id'].nunique():>5} users")

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True)
    )
