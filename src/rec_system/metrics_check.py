"""Check metrics.py against a reference

Two references: the worked example on Lecture 3.1, slide 8, and scikit-learn's ndcg_score.
"""

from collections.abc import Mapping, Sequence

import pandas as pd
from sklearn.metrics import ndcg_score

from rec_system.metrics import ndcg_at_k, precision_at_k, recall_at_k

# Slide 8: Bob's relevant items are {i1, i7}; three top-5 lists with published NDCG@5
SLIDE_8_RELEVANT = {1, 7}
SLIDE_8_LISTS = {
    "A": ([4, 1, 9, 7, 3], 0.65),
    "B": ([1, 7, 9, 4, 3], 1.00),
    "C": ([9, 3, 4, 1, 7], 0.50),
}
SLIDE_8_PRECISION = 0.40
SLIDE_8_RECALL = 1.00


def sklearn_ndcg(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    """NDCG@k from scikit-learn for one ranked list with binary relevance."""
    items = list(dict.fromkeys([*ranked, *sorted(relevant)]))
    y_true = [[1.0 if item in relevant else 0.0 for item in items]]
    # Ranked items get descending scores; unranked relevant items score below all of them
    y_score = [[len(items) - ranked.index(i) if i in ranked else 0.0 for i in items]]
    return float(ndcg_score(y_true, y_score, k=k))


def check_slide_8() -> pd.DataFrame:
    """Our metrics on the slide 8 lists, next to the values printed on the slide."""
    rows = []
    for name, (ranked, slide_ndcg) in SLIDE_8_LISTS.items():
        rows.append(
            {
                "case": f"slide 8 list {name}",
                "precision@5": precision_at_k(ranked, SLIDE_8_RELEVANT, 5),
                "precision@5_reference": SLIDE_8_PRECISION,
                "recall@5": recall_at_k(ranked, SLIDE_8_RELEVANT, 5),
                "recall@5_reference": SLIDE_8_RECALL,
                "ndcg@5": ndcg_at_k(ranked, SLIDE_8_RELEVANT, 5),
                "ndcg@5_reference": slide_ndcg,
            }
        )
    return pd.DataFrame(rows)


def check_against_sklearn(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, set[int]],
    k: int = 10,
    n_users: int = 100,
) -> pd.DataFrame:
    """Our NDCG@k next to scikit-learn's for the first `n_users` users with a full list."""
    rows = []
    for user in sorted(recommendations):
        ranked = list(recommendations[user])[:k]
        if len(ranked) < k or user not in relevant:
            continue
        rows.append(
            {
                "user_id": user,
                f"ndcg@{k}": ndcg_at_k(ranked, relevant[user], k),
                f"ndcg@{k}_sklearn": sklearn_ndcg(ranked, relevant[user], k),
            }
        )
        if len(rows) == n_users:
            break
    return pd.DataFrame(rows)
