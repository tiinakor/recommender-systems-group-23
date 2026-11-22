"""Every Part 2 model behind one interface, with its search space for tuning."""

from collections.abc import Callable
from dataclasses import dataclass

from rec_system.models import als, bpr
from rec_system.models.baseline import item_knn


@dataclass(frozen=True)
class ModelSpec:
    """How to fit a model, how to get recommendations, and which settings to tune.

    fit(user_item_matrix, **params) -> model_data
    recommend(model_data, user_item_matrix, user_id, n) -> list of item ids
    """

    name: str
    fit: Callable
    recommend: Callable
    search_space: dict[str, list]
    uses_seed: bool = False  # True if fit() takes a `seed` argument


MODELS = {
    "item_knn": ModelSpec(
        name="item_knn",
        fit=item_knn.fit_item_knn,
        recommend=item_knn.recommend,
        search_space={
            "metric": list(item_knn.SIMILARITIES),
            "k": [10, 20, 50, 100, 200],
            "aggregation": list(item_knn.AGGREGATIONS),
        },
    ),
    "bpr": ModelSpec(
        name="bpr",
        fit=bpr.fit_bpr,
        recommend=bpr.recommend,
        search_space={
            "k": [16, 32, 64, 128],
            "learning_rate": [0.01, 0.05, 0.1],
            "reg": [0.001, 0.01, 0.05, 0.1],
            "epochs": [20, 40],
        },
        uses_seed=True,
    ),
    "als": ModelSpec(
        name="als",
        fit=als.fit_als,
        recommend=als.recommend,
        search_space={
            "factors": [16, 32, 64, 128],
            "regularization": [1.0, 10.0, 50.0, 100.0, 200.0],
            "alpha": [1.0, 5.0, 10.0, 40.0],
        },
        uses_seed=True,
    ),
}
