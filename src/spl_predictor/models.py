"""The candidate models, from a no-ML baseline up to tree ensembles.

Every candidate answers the same question: given what we knew before a season,
how many points per game will each club earn? Clubs are then ranked by that
number to get a predicted table. The one exception is the original coursework
model, kept here so it can be scored in the same harness as everything else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

CORE_FEATURES = ["prev_ppg", "prev_gd_pg", "promoted"]

# The seven season totals the original notebook used. Only four of them are
# independent (points = 3*wins + draws, goal_diff = goals_for - goals_against,
# wins + draws + losses = games played).
V1_FEATURES = [
    "prev_points", "prev_wins", "prev_draws", "prev_losses",
    "prev_goals_for", "prev_goals_against", "prev_goal_diff",
]


class LastSeason(RegressorMixin, BaseEstimator):
    """Baseline: every club repeats last season's points per game."""

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X).ravel()


class RankClassifier(RegressorMixin, BaseEstimator):
    """The original approach: treat each league position as its own class.

    Returns the negated expected position so that, like every other model,
    a higher prediction means a stronger club.
    """

    def __init__(self, n_neighbors: int = 7):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.model_ = make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=self.n_neighbors)
        ).fit(X, y)
        return self

    def predict(self, X):
        probabilities = self.model_.predict_proba(X)
        return -(probabilities @ self.model_.classes_)


@dataclass(frozen=True)
class Candidate:
    name: str
    features: list[str]
    estimator: BaseEstimator
    target: str = "ppg"

    @property
    def predicts_ppg(self) -> bool:
        return self.target == "ppg"

    def fit(self, train: pd.DataFrame) -> BaseEstimator:
        return self.estimator.fit(train[self.features], train[self.target])


def candidates() -> list[Candidate]:
    """Fresh, unfitted copies of every model in the comparison."""
    return [
        Candidate("Last season", ["prev_ppg"], LastSeason()),
        Candidate(
            "Linear regression",
            CORE_FEATURES,
            # Scaling changes nothing for plain least squares; it is here so
            # the learned weights can be compared with each other.
            make_pipeline(StandardScaler(), LinearRegression()),
        ),
        Candidate(
            "Random forest",
            CORE_FEATURES,
            RandomForestRegressor(n_estimators=300, random_state=SEED),
        ),
        Candidate(
            "Gradient boosting",
            CORE_FEATURES,
            GradientBoostingRegressor(random_state=SEED),
        ),
        Candidate("v1: KNN on rank classes", V1_FEATURES, RankClassifier(), target="position"),
    ]


BASELINE = "Last season"
