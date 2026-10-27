"""Use the chosen model on one matchweek."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from .data import RESULTS
from .features import FEATURES

PICK_NAMES = {"H": "home win", "D": "draw", "A": "away win"}


def predict_week(model, data: pd.DataFrame, season: str, week: int) -> pd.DataFrame:
    """Probabilities for every match of one matchweek.

    The model is trained on every match played before the week's first
    kick-off, and each match's feature uses only results before that match.
    """
    fixtures = data[(data["season"] == season) & (data["week"] == week)]
    train = data[(data["date"] < fixtures["date"].min()) & data["result"].notna()]
    fitted = clone(model).fit(train[FEATURES], train["result"])
    probs = pd.DataFrame(fitted.predict_proba(fixtures[FEATURES]), columns=fitted.classes_,
                         index=fixtures.index)[RESULTS]
    out = fixtures[["date", "home", "away", "gap", "result"]].join(probs.add_prefix("p_"))
    out["pick"] = probs.idxmax(axis=1).map(PICK_NAMES)
    return out.reset_index(drop=True)


def most_likely_winners(week: pd.DataFrame) -> pd.DataFrame:
    """Every club in the week, ranked by its chance of winning its match."""
    home = week[["home", "away", "p_H"]].set_axis(["club", "opponent", "win_chance"], axis=1)
    away = week[["away", "home", "p_A"]].set_axis(["club", "opponent", "win_chance"], axis=1)
    return pd.concat([home, away]).sort_values("win_chance", ascending=False).reset_index(drop=True)
