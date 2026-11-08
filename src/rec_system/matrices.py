
import pandas as pd


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user-item matrix.

    Rows represent users, columns represent items,
    and values represent ratings.
    """

    user_item_matrix = ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0
    )

    return user_item_matrix
