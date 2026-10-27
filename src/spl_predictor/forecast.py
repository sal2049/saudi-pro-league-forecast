"""Use the chosen model: one matchweek at a time, or a whole season."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from .data import RESULTS, league_tables
from .features import FEATURES, add_features


def fit_until(model, data: pd.DataFrame, before: pd.Timestamp):
    """Train on every match played before a date."""
    train = data[(data["date"] < before) & data["result"].notna()]
    return clone(model).fit(train[FEATURES], train["result"])


def predict_week(model, data: pd.DataFrame, season: str, week: int) -> pd.DataFrame:
    """Probabilities for every match of one matchweek.

    The model is trained on everything played before the week's first match,
    and each match's features use only results before that match.
    """
    fixtures = data[(data["season"] == season) & (data["week"] == week)]
    fitted = fit_until(model, data, fixtures["date"].min())
    probs = pd.DataFrame(fitted.predict_proba(fixtures[FEATURES]), columns=fitted.classes_,
                         index=fixtures.index)[RESULTS]
    out = fixtures[["date", "home", "away", "result"]].join(probs.add_prefix("p_"))
    out["pick"] = probs.idxmax(axis=1).map({"H": "home win", "D": "draw", "A": "away win"})
    return out.reset_index(drop=True)


def most_likely_winners(week: pd.DataFrame) -> pd.DataFrame:
    """Every club in the week, ranked by its chance of winning its match."""
    home = week[["home", "away", "p_H"]].set_axis(["club", "opponent", "win_chance"], axis=1).assign(venue="home")
    away = week[["away", "home", "p_A"]].set_axis(["club", "opponent", "win_chance"], axis=1).assign(venue="away")
    return pd.concat([home, away]).sort_values("win_chance", ascending=False).reset_index(drop=True)


def season_table(model, matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Forecast a whole season's table from what was known on its first day.

    The season's results are hidden before the features are built, so every
    club's form stays at its pre-season value for all 34 rounds. Each match
    then gives each side expected points: 3 x P(win) + 1 x P(draw).
    """
    blind = matches.copy()
    in_season = blind["season"] == season
    blind.loc[in_season, ["home_goals", "away_goals", "result"]] = None
    data = add_features(blind)
    fixtures = data[data["season"] == season]
    fitted = fit_until(model, data, fixtures["date"].min())
    p = pd.DataFrame(fitted.predict_proba(fixtures[FEATURES]), columns=fitted.classes_, index=fixtures.index)

    points = pd.concat([
        pd.Series((3 * p["H"] + p["D"]).to_numpy(), index=fixtures["home"].to_numpy()),
        pd.Series((3 * p["A"] + p["D"]).to_numpy(), index=fixtures["away"].to_numpy()),
    ]).groupby(level=0).sum()
    table = points.rename("expected_points").sort_values(ascending=False).to_frame()
    table["predicted_position"] = range(1, len(table) + 1)

    actual = league_tables(matches)
    actual = actual[actual["season"] == season].set_index("team")
    table["points"] = actual["points"]
    table["position"] = actual["position"]
    return table.rename_axis("team").reset_index()
